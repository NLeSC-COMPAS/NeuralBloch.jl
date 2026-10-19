# L2: the I/O contract between the callers and the engines of the model. The
# package checks it before TensorRT gets the data. This file is also the only
# location where the ASCII names of the card (T1, dT1, …) and the public
# Unicode names (T₁, ∂T₁, …) meet.
#
# The contract gives the tensors that this package requires of the engine of
# each target. It is a copy of the upstream model (src/model.py, TARGETS in
# scripts/export_onnx.py): one input and the outputs of the target, no other
# tensors, all Float32. The shapes are in the row-major order of TensorRT. -1
# is the dynamic batch. NTR (readouts for each voxel) and the rank (basis
# coefficients for each voxel) come from the model card.
#   x             input,  (batch, 3): for each voxel [T₁, T₂, B₁] in physical
#                 units (the network calculates the log of T₁ and T₂);
#   signal        output, (batch, NTR, 2): (Re, Im) on the last axis;
#   dT1           output, (batch, NTR, 2): ∂signal/∂T₁ per second;
#   dT2           output, (batch, NTR, 2): ∂signal/∂T₂ per second;
#   coefficients  output, (batch, rank): the basis coefficients of the signal.
# signal_and_derivatives has signal, dT1 and dT2. signal has signal.
# coefficients has coefficients.
#
# The engine file records its tensors (names, dtypes, shapes), but not the
# model that they come from. `Engine` calls `resolve_io` one time for each
# target, when it opens the engine. `resolve_io` compares the tensors with the
# contract and returns the location of each tensor.
#
# The reason for Float32 and these shapes: the row-major (batch, 3) of TensorRT
# is the column-major (3, batch) of Julia. A row-major (batch, NTR, 2) Float32
# tensor has the same bytes as a column-major (NTR, batch) ComplexF32 matrix. A
# row-major (batch, rank) tensor has the same bytes as a column-major (rank,
# batch) Float32 matrix. Thus, the outputs go across without copies. The input
# x is a (3, max) buffer of the engine: `stage!` fills it with three strided
# copies of p.T₁, p.T₂ and p.B₁.

const NAMES = ("T1" => :T₁, "T2" => :T₂, "B1" => :B₁, "signal" => :signal, "dT1" => :∂T₁, "dT2" => :∂T₂,
               "coefficients" => :coefficients)

"""
    publicname(ascii) -> Symbol

The public (Unicode) name for a name of the card or of an engine:
`"dT1"` → `:∂T₁`.
"""
function publicname(ascii::AbstractString)
    i = findfirst(p -> p.first == ascii, NAMES)
    i === nothing && throw(ArgumentError("unknown name $(repr(ascii)); known: $(join(first.(NAMES), ", "))"))
    return NAMES[i].second
end

"""
    cardname(name::Symbol) -> String

The (ASCII) name of the card for a public name: `:∂T₁` → `"dT1"`.
"""
function cardname(name::Symbol)
    i = findfirst(p -> p.second === name, NAMES)
    i === nothing && throw(ArgumentError("unknown name $name; known: $(join(last.(NAMES), ", "))"))
    return NAMES[i].first
end

const INPUT = "x"
const FEATURES = (:T₁, :T₂, :B₁)                   # the rows of x, in order

# The outputs of each target, with their names in the engine.
const OUTPUTS = (signal_and_derivatives = ("signal", "dT1", "dT2"), signal = ("signal",),
                 coefficients = ("coefficients",))

"""
    resolve_io(tensors, target; ntr, rank) -> (; x, outputs...)

Check the engine I/O of one target against the contract, and return the
location of each tensor. `tensors` is what the engine tells about itself
(`TensorRT.tensors`). `target` is `:signal_and_derivatives`, `:signal` or
`:coefficients`. `ntr` and `rank` are the readouts and the basis coefficients
for each voxel, from the model card.

The contract: only the input `x` and the outputs of the target, all Float32.
The shapes are in the row-major order of TensorRT (-1 is the dynamic batch):

| tensor         | direction | shape          | meaning                    | in the targets                       |
|:---------------|:----------|:---------------|:---------------------------|:-------------------------------------|
| `x`            | input     | `(-1, 3)`      | `[T₁, T₂, B₁]` of a voxel  | all                                  |
| `signal`       | output    | `(-1, ntr, 2)` | (Re, Im) per readout       | `signal_and_derivatives`, `signal`   |
| `dT1`          | output    | `(-1, ntr, 2)` | ∂signal/∂T₁ per second     | `signal_and_derivatives`             |
| `dT2`          | output    | `(-1, ntr, 2)` | ∂signal/∂T₂ per second     | `signal_and_derivatives`             |
| `coefficients` | output    | `(-1, rank)`   | basis coefficients of the signal | `coefficients`                 |

Returns the 0-based TensorRT index of `x` and of each output of the target,
with the public names as keys (`dT1` is `∂T₁`). The function finds the
tensors by name, not by position. Thus, the order of the tensors in the ONNX
file has no effect.

A difference from the contract causes a [`BundleError`](@ref): the ONNX file
is not the file that the model card describes. Examples are a missing, extra
or incorrectly named tensor, a dtype that is not Float32, or a different
shape.
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

Return `nothing` if vectors of type `T` can be inputs, otherwise the problem.
All strided GPU vectors of Float32 are applicable, also views with a range:
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

Check `p` against the contract, and return the number of voxels in `p`. The
contract: the properties `T₁`, `T₂` and `B₁` are strided GPU vectors of
Float32 (positive stride) with the same length. If not, the function throws an
`ArgumentError`.
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

The smallest number of columns of a `CuMatrix{T}` with `rows` rows (one
voxel in each column) whose size is a multiple of 256 bytes. If `j - 1` is a
multiple of this number, a column view `A[:, j:k]` starts at a 256-byte
alignment, as TensorRT requires (dev/NOTES.md, S6).
"""
alignment(rows::Integer, ::Type{T}) where {T} = 256 ÷ gcd(256, rows * sizeof(T))

"""
    batch_problem(batch, N, step) -> Union{Nothing,String}

Return `nothing` if an engine for `batch = (; min, opt, max)` voxels in each
call can accept `N` voxels (if `N` is 0, there is nothing to do). Otherwise,
return what to do. `step` is the [`alignment`](@ref) of the outputs, for the
advice about chunks.
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

Return `nothing` if arrays of type `T` can be outputs of element type `E`,
otherwise the problem. Outputs are dense `CuMatrix{E}`, because TensorRT
writes into them directly. `E` is `ComplexF32` for the signal and its
derivatives, and `Float32` for the coefficients.
"""
output_problem(::Type{<:CuArray{E,2}}, ::Type{E}) where {E} = nothing
output_problem(::Type{T}, ::Type{E}) where {T<:CuArray,E} = "a $T; outputs must be CuMatrix{$E}"
output_problem(::Type{T}, ::Type{E}) where {T<:Array,E} = "a $T in CPU memory; outputs must be CuMatrix{$E} on the GPU"
output_problem(::Type{T}, ::Type{E}) where {T<:AbstractArray,E} =
    "a $T; outputs must be dense CuMatrix{$E}: contiguous GPU memory"
output_problem(::Type{T}, ::Type{E}) where {T,E} = "a $T, not an array; outputs must be CuMatrix{$E}"

"""
    layout_problem(size, address, rows, N, E) -> Union{Nothing,String}

Return `nothing` if an output of element type `E`, with `size`, at device
`address`, is applicable to a batch of `N` voxels. Otherwise, return the
problem. The size must be `(rows, N)`. The address must be a multiple of 256
bytes, as TensorRT requires (dev/NOTES.md, S6).
"""
function layout_problem(size::Tuple{Int,Int}, address::UInt, rows::Integer, N::Integer, ::Type{E}) where {E}
    size == (rows, N) || return "of size $size; a batch of $N voxels needs ($rows, $N)"
    address % 256 == 0 && return nothing
    return "not aligned to 256 bytes, as TensorRT requires. Whole CuArrays are; a column view " *
           "A[:, j:k] of a CuMatrix{$E} with $rows rows is when j - 1 is a multiple of $(alignment(rows, E))"
end

"""
    overlap_problem(names, addresses, nbytes) -> Union{Nothing,String}

Return `nothing` if the outputs at `addresses`, each with `nbytes`, do not
overlap. Otherwise, return the two outputs that overlap.
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

The target whose engine calculates the output set `out`: the signal and its
derivatives (in any order), the signal alone, or the coefficients. If `out` is
not one of these sets, the function throws an `ArgumentError`. The keys of
`out`, which are part of its type, give the result. Thus, Julia can calculate
the target, the engine and the layout of a `predict!` call at compile time
(the test "no allocations after warm-up" shows it).
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

The rows and the element type of each output of `target`: `(ntr, ComplexF32)`
for the signal and its derivatives, and `(rank, Float32)` for the
coefficients.
"""
function outputlayout(target::Symbol, ntr::Integer, rank::Integer)
    checktarget(target)
    return target === :coefficients ? (Int(rank), Float32) : (Int(ntr), ComplexF32)
end

address(A::CuArray) = UInt(pointer(A))

"""
    checkoutputs(out, N, rows, E)

Check the output set `out` ([`targetof`](@ref)) against the contract for a
batch of `N` voxels. The contract: dense `CuMatrix{E}`s of size `(rows, N)`
(the [`outputlayout`](@ref) of the target), at a 256-byte alignment, with no
overlap. If not, the function throws an `ArgumentError` that tells what to
change.
"""
function checkoutputs(out::NamedTuple, N::Integer, rows::Integer, ::Type{E}) where {E}
    map(keys(out), values(out)) do name, A
        problem = output_problem(typeof(A), E)
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
        problem = layout_problem(size(A), address(A), rows, N, E)
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
    end
    # Use a new name. If this line assigned `problem`, the closure above would
    # capture a variable that changes, and Julia would box it.
    overlap = overlap_problem(keys(out), map(address, values(out)), rows * N * sizeof(E))
    overlap === nothing || throw(ArgumentError(overlap))
    return nothing
end
