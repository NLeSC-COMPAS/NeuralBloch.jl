# L2: the I/O contract between callers and the model's engine, checked before
# TensorRT sees anything; and the only place where the card's ASCII names
# (T1, dT1, …) meet the public Unicode ones (T₁, ∂T₁, …).
#
# The ONNX (upstream src/model.py) has one input and three outputs:
#   x       Float32 (batch, 3): per voxel [T₁, T₂, B₁] in physical units (the
#           network takes the log of T₁ and T₂ itself);
#   signal  Float32 (batch, NTR, 2), (Re, Im) on the last axis;
#   dT1     the same shape: ∂signal/∂T₁ per second;
#   dT2     the same shape: ∂signal/∂T₂ per second.
# TensorRT's row-major (batch, 3) is Julia's column-major (3, batch), and a
# row-major (batch, NTR, 2) Float32 tensor is byte for byte a column-major
# (NTR, batch) ComplexF32 matrix: inputs and outputs cross without copies.

const NAMES = ("T1" => :T₁, "T2" => :T₂, "B1" => :B₁, "signal" => :signal, "dT1" => :∂T₁, "dT2" => :∂T₂)

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
const OUTPUTS = ("signal", "dT1", "dT2")

"""
    bindings(tensors, ntr) -> (; x, signal, ∂T₁, ∂T₂)

The engine's tensor index of the input and of each output, once the engine's
I/O (`tensors`, see `TensorRT.tensors`) has been checked against the
contract above with `ntr` readouts. A mismatch is a [`BundleError`](@ref):
the ONNX is not the model the card describes.
"""
function bindings(tensors::AbstractVector{TensorRT.TensorInfo}, ntr::Integer)
    expected = "one input x, Float32 (batch, 3) = [T₁, T₂, B₁] per voxel, and outputs signal, dT1 " *
               "and dT2, Float32 (batch, $ntr, 2)"
    bad(what) = throw(BundleError("the engine does not match the model card: $what; expected $expected"))
    index(name, input) = findfirst(t -> t.name == name && t.input == input, tensors)
    function check(i, name, shape)
        t = tensors[i]
        t.eltype === Float32 || bad("$name is $(t.eltype === nothing ? "not Float32" : t.eltype)")
        t.shape == shape || bad("$name has shape $(Tuple(t.shape))")
        return i - 1
    end

    x = index(INPUT, true)
    x === nothing && bad("no input $INPUT")
    outs = map(name -> index(name, false), OUTPUTS)
    for (name, o) in zip(OUTPUTS, outs)
        o === nothing && bad("no output $name")
    end
    for t in tensors
        (t.name == INPUT || t.name in OUTPUTS) || bad("unexpected tensor $(t.name)")
    end
    return (x = check(x, INPUT, [-1, 3]), signal = check(outs[1], "signal", [-1, ntr, 2]),
            ∂T₁ = check(outs[2], "dT1", [-1, ntr, 2]), ∂T₂ = check(outs[3], "dT2", [-1, ntr, 2]))
end

# --- inputs ------------------------------------------------------------------------

"""
    input_problem(T) -> Union{Nothing,String}

`nothing` when vectors of type `T` can be inputs, otherwise what is wrong.
Any GPU vector of Float32 will do, views included: inputs are copied.
"""
input_problem(::Type{<:CUDACore.AnyCuVector{Float32}}) = nothing
input_problem(::Type{T}) where {T<:CUDACore.AnyCuArray} = "a $T; inputs must be GPU vectors of Float32"
input_problem(::Type{T}) where {T<:AbstractArray} =
    "a $T in CPU memory; inputs must be GPU vectors of Float32 (a CuVector{Float32})"
input_problem(::Type{T}) where {T} = "a $T, not an array; inputs must be GPU vectors of Float32"

"""
    checkinputs(p) -> N

The number of voxels in `p`, after checking the contract: properties `T₁`,
`T₂`, `B₁` holding GPU vectors of Float32 of equal length. Otherwise an
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
    return N
end

# --- outputs -----------------------------------------------------------------------

"""
    output_problem(T) -> Union{Nothing,String}

`nothing` when arrays of type `T` can be outputs, otherwise what is wrong:
outputs are dense `CuMatrix{ComplexF32}` (TensorRT writes them in place).
"""
output_problem(::Type{<:CuArray{ComplexF32,2}}) = nothing
output_problem(::Type{T}) where {T<:CuArray} = "a $T; outputs must be CuMatrix{ComplexF32}"
output_problem(::Type{T}) where {T<:Array} = "a $T in CPU memory; outputs must be CuMatrix{ComplexF32} on the GPU"
output_problem(::Type{T}) where {T<:AbstractArray} =
    "a $T; outputs must be dense CuMatrix{ComplexF32}: contiguous GPU memory"
output_problem(::Type{T}) where {T} = "a $T, not an array; outputs must be CuMatrix{ComplexF32}"

"""
    layout_problem(size, address, ntr, N) -> Union{Nothing,String}

`nothing` when an output of `size` at device `address` suits a batch of `N`
voxels, otherwise what is wrong: the size must be `(ntr, N)`, and the
address a multiple of 256 bytes, as TensorRT requires (dev/NOTES.md, S6).
"""
function layout_problem(size::Tuple{Int,Int}, address::UInt, ntr::Integer, N::Integer)
    size == (ntr, N) || return "of size $size; a batch of $N voxels needs ($ntr, $N)"
    address % 256 == 0 && return nothing
    step = 256 ÷ gcd(256, ntr * sizeof(ComplexF32))
    return "not aligned to 256 bytes, as TensorRT requires. Whole CuArrays are; a column view " *
           "A[:, j:k] of a CuMatrix{ComplexF32} with $ntr rows is when j - 1 is a multiple of $step"
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

# (; signal) or (; signal, ∂T₁, ∂T₂), in any order.
outputkeys_ok(keys::Tuple) =
    keys === (:signal,) || (length(keys) == 3 && :signal in keys && :∂T₁ in keys && :∂T₂ in keys)

address(A::CuArray) = UInt(pointer(A))

"""
    checkoutputs(out, ntr, N)

Check `out` against the contract for a batch of `N` voxels: a NamedTuple
`(; signal)` or `(; signal, ∂T₁, ∂T₂)` of dense `(ntr, N)`
`CuMatrix{ComplexF32}`s, 256-byte aligned, not overlapping. Otherwise an
`ArgumentError` saying what to change.
"""
function checkoutputs(out::NamedTuple, ntr::Integer, N::Integer)
    outputkeys_ok(keys(out)) || throw(ArgumentError(
        "out must be (; signal) or (; signal, ∂T₁, ∂T₂), in any order, not a NamedTuple with keys $(keys(out))"))
    map(keys(out), values(out)) do name, A
        problem = output_problem(typeof(A))
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
        problem = layout_problem(size(A), address(A), ntr, N)
        problem === nothing || throw(ArgumentError("out.$name is $problem"))
    end
    problem = overlap_problem(keys(out), map(address, values(out)), ntr * N * sizeof(ComplexF32))
    problem === nothing || throw(ArgumentError(problem))
    return nothing
end

checkoutputs(out, ntr, N) = throw(ArgumentError(
    "out must be a NamedTuple, (; signal) or (; signal, ∂T₁, ∂T₂), not a $(typeof(out))"))
