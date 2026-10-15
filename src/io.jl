# L2: the I/O contract between callers and the model's engines, checked before
# TensorRT sees anything; and the only place where the card's ASCII names
# (T1, dT1, …) meet the public Unicode ones (T₁, ∂T₁, …).
#
# The contract is what this package requires of each target's engine tensors,
# copied from the upstream model (src/model.py, TARGETS in
# scripts/export_onnx.py): one input and the target's outputs, no others, all
# Float32. Shapes are in TensorRT's row-major order, -1 being the dynamic
# batch; NTR (readouts per voxel) and the rank (basis coefficients per voxel)
# come from the model card.
#   x             input,  (batch, 3): per voxel [T₁, T₂, B₁] in physical units
#                 (the network takes the log of T₁ and T₂ itself);
#   signal        output, (batch, NTR, 2): (Re, Im) on the last axis;
#   dT1           output, (batch, NTR, 2): ∂signal/∂T₁ per second;
#   dT2           output, (batch, NTR, 2): ∂signal/∂T₂ per second;
#   coefficients  output, (batch, rank): the signal's basis coefficients.
# signal_and_derivatives has signal, dT1 and dT2; signal has signal;
# coefficients has coefficients.
#
# The engine file records its own tensors (names, dtypes, shapes) but not which
# model they belong to. `resolve_io` compares them with the contract, once, when
# an Engine is built, and returns where each tensor is.
#
# Why Float32 and these shapes: TensorRT's row-major (batch, 3) is Julia's
# column-major (3, batch), a row-major (batch, NTR, 2) Float32 tensor is byte
# for byte a column-major (NTR, batch) ComplexF32 matrix, and a row-major
# (batch, rank) one a column-major (rank, batch) Float32 matrix, so inputs and
# outputs cross without copies.

const NAMES = ("T1" => :T₁, "T2" => :T₂, "B1" => :B₁, "signal" => :signal, "dT1" => :∂T₁, "dT2" => :∂T₂,
               "coefficients" => :coefficients)

"""
    publicname(ascii) -> Symbol

The public (Unicode) name of a card or engine name: `"dT1"` → `:∂T₁`.
"""
function publicname(ascii::AbstractString)
    i = findfirst(p -> p.first == ascii, NAMES)
    i === nothing && throw(ArgumentError("unknown name $(repr(ascii)); known: $(join(first.(NAMES), ", "))"))
    return NAMES[i].second
end

"""
    cardname(name::Symbol) -> String

The card's (ASCII) name for a public name: `:∂T₁` → `"dT1"`.
"""
function cardname(name::Symbol)
    i = findfirst(p -> p.second === name, NAMES)
    i === nothing && throw(ArgumentError("unknown name $name; known: $(join(last.(NAMES), ", "))"))
    return NAMES[i].first
end

const INPUT = "x"
const FEATURES = (:T₁, :T₂, :B₁)                   # the rows of x, in order

# Each target's outputs, by their names in the engine.
const OUTPUTS = (signal_and_derivatives = ("signal", "dT1", "dT2"), signal = ("signal",),
                 coefficients = ("coefficients",))

"""
    resolve_io(tensors, target; ntr, rank) -> (; x, outputs...)

Check one target's engine I/O against the contract and say where each tensor
is. `tensors` is what the engine reports about itself (`TensorRT.tensors`),
`target` one of `:signal_and_derivatives`, `:signal`, `:coefficients`, and
`ntr` and `rank` the readouts and basis coefficients per voxel from the model
card.

The contract: exactly the input `x` and the target's outputs, all Float32,
shapes in TensorRT's row-major order (-1 is the dynamic batch):

| tensor         | direction | shape          | meaning                    | in the targets                       |
|:---------------|:----------|:---------------|:---------------------------|:-------------------------------------|
| `x`            | input     | `(-1, 3)`      | per voxel `[T₁, T₂, B₁]`   | all                                  |
| `signal`       | output    | `(-1, ntr, 2)` | (Re, Im) per readout       | `signal_and_derivatives`, `signal`   |
| `dT1`          | output    | `(-1, ntr, 2)` | ∂signal/∂T₁ per second     | `signal_and_derivatives`             |
| `dT2`          | output    | `(-1, ntr, 2)` | ∂signal/∂T₂ per second     | `signal_and_derivatives`             |
| `coefficients` | output    | `(-1, rank)`   | the signal's basis coefficients | `coefficients`                  |

Returns the 0-based TensorRT index of `x` and of each of the target's outputs,
keyed by public name (`dT1` is `∂T₁`). Tensors are found by name, not by
position, so the result is right whatever order the ONNX lists them in.

Anything that deviates from the contract (a missing, extra or misnamed
tensor, a dtype other than Float32, a different shape) is a
[`BundleError`](@ref): the ONNX is not the one the model card describes.
"""
function resolve_io(tensors::AbstractVector{TensorRT.TensorInfo}, target::Symbol; ntr::Integer, rank::Integer)
    rows, E = outputlayout(target, ntr, rank)
    outputs = OUTPUTS[target]
    outshape = E <: Complex ? [-1, rows, 2] : [-1, rows]     # TensorRT has no complex type: (Re, Im) last
    listing = length(outputs) == 1 ? "output $(only(outputs))" : "outputs " * join(outputs, ", ", " and ")
    expected = "one input x, Float32 (batch, 3) = [T₁, T₂, B₁] per voxel, and $listing, Float32 " *
               "(batch, $(join(outshape[2:end], ", ")))"
    bad(what) = throw(BundleError("the $target engine does not match the model card: $what; expected $expected"))
    index(name, input) = findfirst(t -> t.name == name && t.input == input, tensors)
    function check(i, name, shape)
        t = tensors[i]
        t.eltype === Float32 || bad("$name is $(t.eltype === nothing ? "not Float32" : t.eltype)")
        t.shape == shape || bad("$name has shape $(Tuple(t.shape))")
        return i - 1
    end

    x = index(INPUT, true)
    x === nothing && bad("no input $INPUT")
    outs = map(name -> index(name, false), outputs)
    for (name, o) in zip(outputs, outs)
        o === nothing && bad("no output $name")
    end
    for t in tensors
        (t.name == INPUT || t.name in outputs) || bad("unexpected tensor $(t.name)")
    end
    indices = map((name, o) -> check(o, name, outshape), outputs, outs)
    return NamedTuple{(:x, map(publicname, outputs)...)}((check(x, INPUT, [-1, 3]), indices...))
end

# --- inputs ------------------------------------------------------------------------

"""
    input_problem(T) -> Union{Nothing,String}

`nothing` when vectors of type `T` can be inputs, otherwise what is wrong.
Any strided GPU vector of Float32 will do, views with a range included:
predict! copies them with strided copies.
"""
input_problem(::Type{<:CUDACore.StridedCuVector{Float32}}) = nothing
input_problem(::Type{T}) where {T<:CUDACore.AnyCuArray} =
    "a $T; inputs must be strided GPU vectors of Float32: a CuVector{Float32}, or a view of one with a range"
input_problem(::Type{T}) where {T<:AbstractArray} =
    "a $T in CPU memory; inputs must be GPU vectors of Float32 (a CuVector{Float32})"
input_problem(::Type{T}) where {T} = "a $T, not an array; inputs must be GPU vectors of Float32"

"""
    checkinputs(p) -> N

The number of voxels in `p`, after checking the contract: properties `T₁`,
`T₂`, `B₁` holding strided GPU vectors of Float32 (positive stride) of equal
length. Otherwise an `ArgumentError`.
"""
function checkinputs(p)
    for name in FEATURES
        hasproperty(p, name) || throw(ArgumentError(
            "p has no property $name: predict! reads p.T₁, p.T₂ and p.B₁, GPU vectors of Float32 in " *
            "physical units (T₁, T₂ in seconds, B₁ relative)"))
    end
    T₁, T₂, B₁ = p.T₁, p.T₂, p.B₁
    for (name, problem) in ((:T₁, input_problem(typeof(T₁))), (:T₂, input_problem(typeof(T₂))),
                            (:B₁, input_problem(typeof(B₁))))
        problem === nothing || throw(ArgumentError("p.$name is $problem"))
    end
    N = length(T₁)
    length(T₂) == length(B₁) == N || throw(ArgumentError(
        "p.T₁, p.T₂ and p.B₁ have lengths $N, $(length(T₂)) and $(length(B₁)); they must be equal"))
    for (name, s) in ((:T₁, stride(T₁, 1)), (:T₂, stride(T₂, 1)), (:B₁, stride(B₁, 1)))
        s ≥ 1 || throw(ArgumentError("p.$name has stride $s; inputs must have a positive stride (copy it first)"))
    end
    return N
end

"""
    alignment(rows, T) -> Int

The voxels per 256-byte step of a `CuMatrix{T}` with `rows` rows, a voxel per
column: a column view `A[:, j:k]` starts 256-byte aligned, as TensorRT
requires (dev/NOTES.md, S6), when `j - 1` is a multiple of it.
"""
alignment(rows::Integer, ::Type{T}) where {T} = 256 ÷ gcd(256, rows * sizeof(T))

"""
    batch_problem(batch, N, step) -> Union{Nothing,String}

`nothing` when an engine accepting `batch = (; min, opt, max)` voxels per
call can take `N` voxels (0 is nothing to do), otherwise what to do instead.
`step` is the outputs' [`alignment`](@ref), for the advice on chunking.
"""
function batch_problem(batch::NamedTuple, N::Integer, step::Integer)
    (N == 0 || batch.min ≤ N ≤ batch.max) && return nothing
    what = "a batch of $N voxels is outside this engine's range, $(batch.min) to $(batch.max) voxels. "
    N < batch.min && return what * "Use an engine built for smaller batches: " *
                            "Engine(model; batch = (min, opt, max)) with min ≤ $N."
    return what * "Split it into chunks (column views of the outputs; chunk boundaries at multiples of " *
           "$step voxels keep them aligned), or use an engine built for it: " *
           "Engine(model; batch = (min, opt, max)) with max ≥ $N."
end

# --- outputs -----------------------------------------------------------------------

"""
    output_problem(T, E) -> Union{Nothing,String}

`nothing` when arrays of type `T` can be outputs of element type `E`,
otherwise what is wrong: outputs are dense `CuMatrix{E}` (TensorRT writes
them in place), `ComplexF32` for the signal and its derivatives and `Float32`
for the coefficients.
"""
output_problem(::Type{<:CuArray{E,2}}, ::Type{E}) where {E} = nothing
output_problem(::Type{T}, ::Type{E}) where {T<:CuArray,E} = "a $T; outputs must be CuMatrix{$E}"
output_problem(::Type{T}, ::Type{E}) where {T<:Array,E} = "a $T in CPU memory; outputs must be CuMatrix{$E} on the GPU"
output_problem(::Type{T}, ::Type{E}) where {T<:AbstractArray,E} =
    "a $T; outputs must be dense CuMatrix{$E}: contiguous GPU memory"
output_problem(::Type{T}, ::Type{E}) where {T,E} = "a $T, not an array; outputs must be CuMatrix{$E}"

"""
    layout_problem(size, address, rows, N, E) -> Union{Nothing,String}

`nothing` when an output of `size` at device `address`, of element type
`E`, suits a batch of `N` voxels, otherwise what is wrong: the size must be
`(rows, N)`, and the address a multiple of 256 bytes, as TensorRT requires
(dev/NOTES.md, S6).
"""
function layout_problem(size::Tuple{Int,Int}, address::UInt, rows::Integer, N::Integer, ::Type{E}) where {E}
    size == (rows, N) || return "of size $size; a batch of $N voxels needs ($rows, $N)"
    address % 256 == 0 && return nothing
    return "not aligned to 256 bytes, as TensorRT requires. Whole CuArrays are; a column view " *
           "A[:, j:k] of a CuMatrix{$E} with $rows rows is when j - 1 is a multiple of $(alignment(rows, E))"
end

"""
    overlap_problem(names, addresses, nbytes) -> Union{Nothing,String}

`nothing` when outputs of `nbytes` each, at `addresses`, do not overlap;
otherwise which two do.
"""
function overlap_problem(names::Tuple, addresses::Tuple, nbytes::Integer)
    for i in eachindex(addresses), j in i+1:lastindex(addresses)
        if addresses[i] < addresses[j] + nbytes && addresses[j] < addresses[i] + nbytes
            return "out.$(names[i]) and out.$(names[j]) overlap in memory; every output needs its own"
        end
    end
    return nothing
end

const OUTPUT_SETS = "(; signal, ∂T₁, ∂T₂) in any order, (; signal) or (; coefficients)"

"""
    targetof(out) -> Symbol

The target whose engine computes the output set `out`: the signal and its
derivatives (in any order), the signal alone, or the coefficients. An
`ArgumentError` unless `out` is one of these sets. Decided by the type of
`out`, so `predict!` knows its target, engine and layout statically.
"""
targetof(::NamedTuple{(:signal,)}) = :signal
targetof(::NamedTuple{(:coefficients,)}) = :coefficients
function targetof(out::NamedTuple)
    names = keys(out)
    length(names) == 3 && :signal in names && :∂T₁ in names && :∂T₂ in names && return :signal_and_derivatives
    throw(ArgumentError("out must be $OUTPUT_SETS, not a NamedTuple with keys $names"))
end
targetof(out) = throw(ArgumentError("out must be a NamedTuple, $OUTPUT_SETS, not a $(typeof(out))"))

"""
    outputlayout(target, ntr, rank) -> (rows, E)

The rows and element type of each of `target`'s outputs: `(ntr, ComplexF32)`
for the signal and its derivatives, `(rank, Float32)` for the coefficients.
"""
function outputlayout(target::Symbol, ntr::Integer, rank::Integer)
    checktarget(target)
    return target === :coefficients ? (Int(rank), Float32) : (Int(ntr), ComplexF32)
end

address(A::CuArray) = UInt(pointer(A))

"""
    checkoutputs(out, N, rows, E)

Check `out`, an output set ([`targetof`](@ref)), against the contract for a
batch of `N` voxels: dense `CuMatrix{E}`s of size `(rows, N)` (its target's
[`outputlayout`](@ref)), 256-byte aligned, not overlapping. Otherwise an
`ArgumentError` saying what to change.
"""
function checkoutputs(out::NamedTuple, N::Integer, rows::Integer, ::Type{E}) where {E}
    map(keys(out), values(out)) do name, A
        problem = output_problem(typeof(A), E)
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
        problem = layout_problem(size(A), address(A), rows, N, E)
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
    end
    # (A name of its own: reassigning a variable the closure above uses would box it.)
    overlap = overlap_problem(keys(out), map(address, values(out)), rows * N * sizeof(E))
    overlap === nothing || throw(ArgumentError(overlap))
    return nothing
end
