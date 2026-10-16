# L2: the model bundle — an ONNX model per target, the card, and the RF tables
# of the sequence the network was trained on. NeuralBloch's export_bundle
# writes the bundle and NeuralBloch publishes it as a release, which Pkg
# installs with this package as an artifact (Artifacts.toml); nothing in it
# is edited here. The card (model_card.toml) uses ASCII names (T1, dT1, …);
# io.jl converts them for the public API.

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

# The card's file in a bundle, and the MAJOR of the card schema this package
# reads, as NeuralBloch's export_bundle writes them (scripts/export_bundle.py,
# CARD and SCHEMA_VERSION). The card's schema_version is "MAJOR.MINOR": a
# newer MINOR only adds fields, which are ignored here, and another MAJOR
# changes the card, so it is refused.
const CARD = "model_card.toml"
const SCHEMA_MAJOR = 1

# What the model can compute, each from its own ONNX model and engine:
#   signal_and_derivatives  signal, ∂T₁, ∂T₂   (upstream FCModel.forward)
#   signal                  signal             (FCModel.forward_signal)
#   coefficients            coefficients       (FCModel.forward_coefficients)
const TARGETS = (:signal_and_derivatives, :signal, :coefficients)

# A target's name, or an ArgumentError.
checktarget(target::Symbol) = target in TARGETS || throw(ArgumentError(
    "target must be $(join(repr.(TARGETS), ", ", " or ")), not $(repr(target))"))

# The domain constraints checkdomain implements.
const CONSTRAINTS = ("T2 < T1",)

# How the training targets' derivatives may have been computed.
const DERIVATIVE_METHODS = ("forward sensitivity", "forward difference")

# A family is named <sequence><ntr>-rank<rank>, optionally with a qualifier:
# lowercase words joined by hyphens, exactly one of them rank<N>.
const FAMILY_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

# A version is a semantic version without build metadata, as NeuralBloch's
# export_bundle writes it (scripts/export_bundle.py, SEMVER): MAJOR.MINOR.PATCH,
# optionally with a pre-release such as -beta.1. VersionNumber then orders
# versions as semantic versions; on its own it would also take "1", "v0.1.0"
# or "01.0.0".
const PRERELEASE_PART = raw"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
const SEMVER = Regex(raw"\A(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)" *
                     "(?:-$PRERELEASE_PART(?:\\.$PRERELEASE_PART)*)?" * raw"\z")

"""
    Model(dir = the fisp600-rank16 artifact)

A NeuralBloch model bundle: `dir/model_card.toml` and the files it names, an
ONNX model per target and the RF tables. Without `dir`, the `fisp600-rank16`
bundle that this package's Artifacts.toml binds, which Pkg downloads from
NeuralBloch's release when it installs the package; any other bundle, such as
your own export, is read from its directory. Loading checks all of it: the
card's schema version (any 1.x), every field present and well-typed, each
ONNX's sha256 the one the card pins, and the RF tables with one entry per
readout. Any problem is a [`BundleError`](@ref).

The fields hold the card: `family` and `version` (the model's name: a family
such as `fisp600-rank16` and a semantic version such as `v"0.1.0-beta.1"`),
`ntr` (readouts per voxel), `rank` (signal basis coefficients per voxel),
`onnx` and `sha256` (per target: `signal_and_derivatives`, `signal`,
`coefficients`), `domain` (training ranges of T1, T2 and B1), `constraint`,
`derivatives` (how the training targets were computed: the method, and the
steps of forward differences), `sequence` (the FISP3D sequence, RF tables in
degrees) and `provenance`.
"""
struct Model
    dir::String
    family::String
    version::VersionNumber
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
describe(::Type{VersionNumber}) =
    "a semantic version MAJOR.MINOR.PATCH[-PRERELEASE] without build metadata, e.g. \"0.1.0-beta.1\""

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
    T === VersionNumber && v isa AbstractString && occursin(SEMVER, v) && return VersionNumber(v)
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
        throw(BundleError("$file has $(length(values)) values; [model] ntr is $ntr (see $CARD)"))
    return values
end

# The rank a family name states, or nothing if it does not follow the pattern.
function familyrank(family)
    occursin(FAMILY_PATTERN, family) || return nothing
    ranks = [m[1] for m in (match(r"^rank([1-9][0-9]*)$", w) for w in split(family, '-')) if m !== nothing]
    return length(ranks) == 1 ? parse(Int, ranks[1]) : nothing
end

function Model(dir::AbstractString = artifact"neuralbloch_fisp600_rank16")
    dir = abspath(dir)
    path = joinpath(dir, CARD)
    if !isfile(path)
        isfile(joinpath(dir, "card.toml")) && throw(BundleError(
            "$dir has card.toml, a card from before schema version 1.0, but no $CARD, which this " *
            "NeuralBloch reads. Use a bundle written by a newer NeuralBloch export_bundle"))
        throw(BundleError("$dir has no $CARD: it is not a model bundle"))
    end
    card = try
        TOML.parsefile(path)
    catch err
        err isa TOML.ParserError || rethrow()
        throw(BundleError("$path is not valid TOML: " * sprint(showerror, err)))
    end
    need(T, table, key) = field(card, path, table, key, T)
    positive(T, table, key) = (v = need(T, table, key);
                               v > 0 || throw(BundleError("$path: [$table] $key must be positive, not $v")); v)

    schema = get(card, "schema_version", nothing)
    schema === nothing && throw(BundleError("$path has no schema_version; this NeuralBloch reads schema $SCHEMA_MAJOR.x"))
    parts = schema isa AbstractString ? match(r"\A(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\z", schema) : nothing
    parts === nothing && throw(BundleError(
        "$path: schema_version must be \"MAJOR.MINOR\", e.g. \"$SCHEMA_MAJOR.0\", not $(repr(schema))"))
    parse(Int, parts[1]) == SCHEMA_MAJOR || throw(BundleError(
        "$path has schema_version $schema; this NeuralBloch reads schema $SCHEMA_MAJOR.x"))

    family = need(String, "model", "family")
    version = need(VersionNumber, "model", "version")
    ntr = positive(Int, "model", "ntr")
    rank = positive(Int, "model", "rank")
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
    # A step, given exactly when the method is forward differences.
    function fdstep(key)
        method == "forward difference" && return positive(Float64, "derivatives", key)
        haskey(card["derivatives"], key) && throw(BundleError(
            "$path: [derivatives] $key is given, but steps apply only to forward differences, not $method"))
        return nothing
    end
    derivatives = (; method, dT1_step = fdstep("dT1_step"), dT2_step = fdstep("dT2_step"))

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

"""
    contract_problems(current::Model, new::Model) -> Vector{String}

What keeps bundle `new` from replacing `current` as a later version of the
same model family; empty when nothing does. Within a family, callers rely on
the readouts, the rank, the sequence (RF tables included) and the domain
constraint staying the same, on the training domain staying the same or
growing, and on the version going up in semantic-version order
(0.1.0-beta.1 < 0.1.0-beta.2 < 0.1.0 < 0.1.1). Two different families share
no contract, so `new` of another family has no problems here.
"""
function contract_problems(current::Model, new::Model)
    current.family == new.family || return String[]
    problems = String[]
    new.version > current.version ||
        push!(problems, "version $(new.version) does not follow version $(current.version)")
    for key in (:ntr, :rank, :constraint)
        a, b = getfield(current, key), getfield(new, key)
        a == b || push!(problems, "$key $(repr(a)) → $(repr(b))")
    end
    for key in keys(current.sequence)
        a, b = current.sequence[key], new.sequence[key]
        a == b || push!(problems, key in (:rf_angle, :rf_phase) ? "the $key table differs" : "sequence $key $(repr(a)) → $(repr(b))")
    end
    for (name, (lo, hi)) in pairs(current.domain)
        newlo, newhi = new.domain[name]
        newlo ≤ lo && hi ≤ newhi ||
            push!(problems, "domain $name [$lo, $hi] → [$newlo, $newhi] does not contain the previous range")
    end
    return problems
end
