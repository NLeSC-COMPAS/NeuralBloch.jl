# L2: the model bundle — an ONNX model per target, the card, and the RF tables
# of the sequence the network was trained on. NeuralBloch's export_bundle
# writes the bundle; nothing in it is edited here. The card (models/card.toml)
# uses ASCII names (T1, dT1, …); io.jl converts them for the public API.

"""
    BundleError(msg)

The model bundle is incomplete or inconsistent: its card lacks a field or has
a malformed one, a file it names is missing, or an ONNX model is not the one
the card pins. The message names the file to fix.
"""
struct BundleError <: Exception
    msg::String
end

Base.showerror(io::IO, e::BundleError) = print(io, "BundleError: ", e.msg)

const Range = Tuple{Float64,Float64}

# The card format this package reads: the one NeuralBloch's export_bundle
# writes (src/bundle.py, FORMAT).
const FORMAT = 2

# What the model can compute, each from its own ONNX model and engine:
#   signal_and_derivatives  signal, ∂T₁, ∂T₂   (upstream FCModel.forward)
#   signal                  signal             (FCModel.forward_signal)
#   coefficients            coefficients       (FCModel.forward_coefficients)
const TARGETS = (:signal_and_derivatives, :signal, :coefficients)

# The domain constraints checkdomain implements.
const CONSTRAINTS = ("T2 < T1",)

# How the training targets' derivatives may have been computed.
const DERIVATIVE_METHODS = ("forward sensitivity", "forward difference")

# A family is named <sequence><ntr>-rank<rank>, optionally with a qualifier:
# lowercase words joined by hyphens, exactly one of them rank<N>.
const FAMILY_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

"""
    Model(dir = the bundled model)

A NeuralBloch model bundle: `dir/card.toml` and the files it names, an ONNX
model per target and the RF tables. Loading checks all of it: the card's
format, every field present and well-typed, each ONNX's sha256 the one the
card pins, and the RF tables with one entry per readout. Any problem is a
[`BundleError`](@ref).

The fields hold the card: `family` and `version` (the model's name, e.g.
`fisp600-rank16` version 1), `ntr` (readouts per voxel), `rank` (signal basis
coefficients per voxel), `onnx` and `sha256` (per target:
`signal_and_derivatives`, `signal`, `coefficients`), `domain` (training
ranges of T1, T2 and B1), `constraint`, `derivatives` (how the training
targets were computed: the method, and the steps of forward differences),
`sequence` (the FISP3D sequence, RF tables in degrees) and `provenance`.
"""
struct Model
    dir::String
    family::String
    version::Int
    ntr::Int
    rank::Int
    onnx::NamedTuple{TARGETS,NTuple{3,String}}
    sha256::NamedTuple{TARGETS,NTuple{3,String}}
    domain::@NamedTuple{T1::Range, T2::Range, B1::Range}
    constraint::String
    derivatives::@NamedTuple{method::String, dT1_step::Union{Nothing,Float64}, dT2_step::Union{Nothing,Float64}}
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

# card[table][key] as a T, or a BundleError naming the field. A dotted table
# name is a nested table: "model.signal" is card["model"]["signal"].
function field(card, path, table, key, ::Type{T}) where {T}
    t = card
    for name in split(table, '.')
        t = t isa AbstractDict ? get(t, name, nothing) : nothing
    end
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

# The rank a family name states, or nothing if it does not follow the pattern.
function familyrank(family)
    occursin(FAMILY_PATTERN, family) || return nothing
    ranks = [m[1] for m in (match(r"^rank([1-9][0-9]*)$", w) for w in split(family, '-')) if m !== nothing]
    return length(ranks) == 1 ? parse(Int, ranks[1]) : nothing
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
    positive(table, key) = (v = need(Int, table, key);
                            v ≥ 1 || throw(BundleError("$path: [$table] $key must be positive, not $v")); v)
    positivestep(key) = (v = need(Float64, "derivatives", key);
                         v > 0 || throw(BundleError("$path: [derivatives] $key must be positive, not $v")); v)

    format = get(card, "format", nothing)
    format === nothing && throw(BundleError(
        "$path has no format: it predates card format $FORMAT, which this NeuralBloch reads. " *
        "Use a bundle written by NeuralBloch's export_bundle"))
    format == FORMAT || throw(BundleError(
        "$path has card format $(repr(format)); this NeuralBloch reads format $FORMAT"))

    family = need(String, "model", "family")
    version = positive("model", "version")
    ntr = positive("model", "ntr")
    rank = positive("model", "rank")
    stated = familyrank(family)
    stated === nothing && throw(BundleError(
        "$path: [model] family $(repr(family)) must be lowercase words joined by hyphens, exactly one " *
        "of them rank<N>: <sequence><ntr>-rank<rank>, e.g. fisp600-rank16"))
    stated == rank || throw(BundleError(
        "$path: [model] family $(repr(family)) states rank $stated, but [model] rank is $rank"))

    pinned = map(TARGETS) do target
        table = "model.$target"
        sha = need(String, table, "sha256")
        occursin(r"^[0-9a-f]{64}$", sha) ||
            throw(BundleError("$path: [$table] sha256 must be 64 hexadecimal digits, not $(repr(sha))"))
        file = bundlefile(dir, path, table, "onnx", need(String, table, "onnx"))
        actual = bytes2hex(open(SHA.sha256, file))
        actual == sha || throw(BundleError(
            "$file has sha256 $actual, but $path pins $sha for $target: it is not the model the card describes"))
        (file, sha)
    end
    onnx = NamedTuple{TARGETS}(map(first, pinned))
    sha256 = NamedTuple{TARGETS}(map(last, pinned))

    domain = (T1 = need(Range, "domain", "T1"), T2 = need(Range, "domain", "T2"), B1 = need(Range, "domain", "B1"))
    constraint = need(String, "domain", "constraint")
    constraint in CONSTRAINTS || throw(BundleError(
        "$path: [domain] has an unsupported constraint $(repr(constraint)); supported: " *
        join(repr.(CONSTRAINTS), ", ")))

    method = need(String, "derivatives", "method")
    method in DERIVATIVE_METHODS || throw(BundleError(
        "$path: [derivatives] method $(repr(method)) is not one of " * join(repr.(DERIVATIVE_METHODS), ", ")))
    steps = if method == "forward difference"
        (positivestep("dT1_step"), positivestep("dT2_step"))
    else
        for key in ("dT1_step", "dT2_step")
            haskey(card["derivatives"], key) && throw(BundleError(
                "$path: [derivatives] $key is given, but steps apply only to forward differences, not $method"))
        end
        (nothing, nothing)
    end
    derivatives = (; method, dT1_step = steps[1], dT2_step = steps[2])

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
    return Model(dir, family, version, ntr, rank, onnx, sha256, domain, constraint, derivatives, sequence,
                 provenance)
end

Base.show(io::IO, m::Model) =
    print(io, "Model(", repr(m.dir), "; ", m.family, " v", m.version, ", ntr = ", m.ntr, ", rank = ", m.rank, ")")
