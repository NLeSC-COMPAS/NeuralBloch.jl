# L2: the model bundle — the ONNX, its card, and the RF tables of the sequence
# the network was trained on. The card (models/card.toml) uses ASCII names
# (T1, dT1, …); io.jl converts them for the public API.

"""
    BundleError(msg)

The model bundle is incomplete or inconsistent: its card lacks a field or has
a malformed one, a file it names is missing, or the ONNX is not the one the
card pins. The message names the file to fix.
"""
struct BundleError <: Exception
    msg::String
end

Base.showerror(io::IO, e::BundleError) = print(io, "BundleError: ", e.msg)

const Range = Tuple{Float64,Float64}

# The domain constraints checkdomain implements.
const CONSTRAINTS = ("T2 < T1",)

"""
    Model(dir = the bundled model)

A NeuralBloch model bundle: `dir/card.toml` and the files it names, the
ONNX and the RF tables. Loading checks all of it: every card field is
present and well-typed, the ONNX's sha256 is the one the card pins, and the
RF tables have one entry per readout. Any problem is a [`BundleError`](@ref).

The fields hold the card: `onnx` (path), `sha256`, `ntr` (readouts per
voxel), `domain` (training ranges of T1, T2 and B1), `constraint`,
`derivatives` (how the training targets were computed), `sequence` (the
FISP3D sequence, RF tables in degrees) and `provenance`.
"""
struct Model
    dir::String
    onnx::String
    sha256::String
    ntr::Int
    domain::@NamedTuple{T1::Range, T2::Range, B1::Range}
    constraint::String
    derivatives::@NamedTuple{method::String, dT1_step::Float64, dT2_step::Float64}
    sequence::@NamedTuple{kind::String, TR::Float64, TE::Float64, TI::Float64, TW::Float64,
                          max_state::Int, repetitions::Int, inversion_prepulse::Bool,
                          wait_spoiling::Bool, undersampling_factor::Int, delta_k_spoil::Float64,
                          rf_angle::Vector{Float64}, rf_phase::Vector{Float64}}
    provenance::Dict{String,String}
end

describe(::Type{Float64}) = "a number"
describe(::Type{Int}) = "an integer"
describe(::Type{Bool}) = "true or false"
describe(::Type{String}) = "a string"
describe(::Type{Range}) = "two numbers [low, high] with low < high"

isnumber(x) = x isa Real && !(x isa Bool)

# card[table][key] as a T, or a BundleError naming the field.
function field(card, path, table, key, ::Type{T}) where {T}
    t = get(card, table, nothing)
    t isa AbstractDict || throw(BundleError("$path: table [$table] is missing"))
    haskey(t, key) || throw(BundleError("$path: [$table] $key is missing"))
    v = t[key]
    T === Float64 && isnumber(v) && return Float64(v)
    T === Int && v isa Integer && !(v isa Bool) && return Int(v)
    T === Bool && v isa Bool && return v
    T === String && v isa AbstractString && return String(v)
    if T === Range && v isa AbstractVector && length(v) == 2 && all(isnumber, v) && v[1] < v[2]
        return (Float64(v[1]), Float64(v[2]))
    end
    throw(BundleError("$path: [$table] $key must be $(describe(T)), not $(repr(v))"))
end

# A file the card names, relative to the bundle.
function bundlefile(dir, path, table, key, name)
    file = joinpath(dir, name)
    isfile(file) || throw(BundleError("$path: [$table] $key names $name, which $dir does not have"))
    return file
end

# One number per line, `ntr` of them.
function rftable(file, ntr)
    values = Float64[]
    for (i, line) in enumerate(eachline(file))
        isempty(strip(line)) && continue
        v = tryparse(Float64, line)
        v === nothing && throw(BundleError("$file:$i: not a number: $(repr(line))"))
        push!(values, v)
    end
    length(values) == ntr ||
        throw(BundleError("$file has $(length(values)) values; [model] ntr is $ntr (see card.toml)"))
    return values
end

function Model(dir::AbstractString = joinpath(pkgdir(@__MODULE__), "models"))
    dir = abspath(dir)
    path = joinpath(dir, "card.toml")
    isfile(path) || throw(BundleError("$dir has no card.toml: it is not a model bundle"))
    card = try
        TOML.parsefile(path)
    catch err
        err isa TOML.ParserError || rethrow()
        throw(BundleError("$path is not valid TOML: " * sprint(showerror, err)))
    end
    need(T, table, key) = field(card, path, table, key, T)

    sha = need(String, "model", "sha256")
    occursin(r"^[0-9a-f]{64}$", sha) ||
        throw(BundleError("$path: [model] sha256 must be 64 hexadecimal digits, not $(repr(sha))"))
    onnx = bundlefile(dir, path, "model", "onnx", need(String, "model", "onnx"))
    actual = bytes2hex(open(SHA.sha256, onnx))
    actual == sha || throw(BundleError(
        "$onnx has sha256 $actual, but $path pins $sha: it is not the model the card describes"))
    ntr = need(Int, "model", "ntr")
    ntr ≥ 1 || throw(BundleError("$path: [model] ntr must be positive, not $ntr"))

    domain = (T1 = need(Range, "domain", "T1"), T2 = need(Range, "domain", "T2"), B1 = need(Range, "domain", "B1"))
    constraint = need(String, "domain", "constraint")
    constraint in CONSTRAINTS || throw(BundleError(
        "$path: [domain] has an unsupported constraint $(repr(constraint)); supported: " *
        join(repr.(CONSTRAINTS), ", ")))
    derivatives = (method = need(String, "derivatives", "method"),
                   dT1_step = need(Float64, "derivatives", "dT1_step"),
                   dT2_step = need(Float64, "derivatives", "dT2_step"))
    rf(key) = rftable(bundlefile(dir, path, "sequence", key, need(String, "sequence", key)), ntr)
    sequence = (kind = need(String, "sequence", "kind"),
                TR = need(Float64, "sequence", "TR"), TE = need(Float64, "sequence", "TE"),
                TI = need(Float64, "sequence", "TI"), TW = need(Float64, "sequence", "TW"),
                max_state = need(Int, "sequence", "max_state"),
                repetitions = need(Int, "sequence", "repetitions"),
                inversion_prepulse = need(Bool, "sequence", "inversion_prepulse"),
                wait_spoiling = need(Bool, "sequence", "wait_spoiling"),
                undersampling_factor = need(Int, "sequence", "undersampling_factor"),
                delta_k_spoil = need(Float64, "sequence", "delta_k_spoil"),
                rf_angle = rf("rf_angle"), rf_phase = rf("rf_phase"))
    need(String, "provenance", "upstream")
    provenance = Dict{String,String}(k => string(v) for (k, v) in card["provenance"])
    return Model(dir, onnx, sha, ntr, domain, constraint, derivatives, sequence, provenance)
end

Base.show(io::IO, m::Model) =
    print(io, "Model(", repr(m.dir), "; ntr = ", m.ntr, ", sha256 = ", m.sha256[1:12], "…)")
