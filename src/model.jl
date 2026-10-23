# L2: the model bundle. The bundle has an ONNX model for each target, the card,
# and the RF tables of the sequence that the network was trained on. The
# export_bundle script of NeuralBloch writes the bundle, and NeuralBloch
# publishes it as a release. Pkg installs the release with this package as an
# artifact (Artifacts.toml). This package does not change the bundle. The card
# (model_card.toml) uses ASCII names (T1, dT1, …). io.jl changes them to the
# names of the public API.

"""
    BundleError(msg)

The model bundle is not complete or not consistent. Possible causes: a field
of the card is missing or incorrect, a file that the card names is missing, or
an ONNX model is not the model that the card pins. [`Model`](@ref) throws it
for the card and its files. [`Engine`](@ref) throws it if the inputs or
outputs of an engine do not agree with the card. The message gives the file
or the target to correct.
"""
struct BundleError <: Exception
    msg::String
end

Base.showerror(io::IO, e::BundleError) = print(io, "BundleError: ", e.msg)

const Range = Tuple{Float64,Float64}

# The file name of the card in a bundle, and the MAJOR of the card schema that
# this package reads. The export_bundle script of NeuralBloch uses the same
# values (scripts/export_bundle.py, CARD and SCHEMA_VERSION). The
# schema_version of a card is "MAJOR.MINOR". A newer MINOR only adds fields,
# and this package ignores them. A different MAJOR changes the card, and this
# package refuses it.
const CARD = "model_card.toml"
const SCHEMA_MAJOR = 1

# The targets: what the model can calculate. Each target has its own ONNX model
# and engine.
#   signal_and_derivatives  signal, ∂T₁, ∂T₂   (upstream FCModel.forward)
#   signal                  signal             (FCModel.forward_signal)
#   coefficients            coefficients       (FCModel.forward_coefficients)
const TARGETS = (:signal_and_derivatives, :signal, :coefficients)

# Throw an ArgumentError if `target` is not a target.
checktarget(target::Symbol) = target in TARGETS || throw(ArgumentError(
    "target must be $(join(repr.(TARGETS), ", ", " or ")), not $(repr(target))"))

# The domain constraints that checkdomain applies.
const CONSTRAINTS = ("T2 < T1",)

# The methods that can calculate the derivatives of the training targets.
const DERIVATIVE_METHODS = ("forward sensitivity", "forward difference")

# The name of a family: lowercase words (letters and digits) with hyphens
# between them. Exactly one word is rank<N>, and N must be the rank of the
# card. Model checks only these rules. By convention, the name is
# <sequence><ntr>-rank<rank>, with an optional qualifier, but Model does not
# compare <ntr> with the ntr of the card.
const FAMILY_PATTERN = r"\A[a-z0-9]+(-[a-z0-9]+)*\z"

# A version is a semantic version without build metadata: MAJOR.MINOR.PATCH,
# with an optional pre-release such as -beta.1. The export_bundle script of
# NeuralBloch uses the same pattern (scripts/export_bundle.py, SEMVER). After
# this check, VersionNumber sorts the versions as semantic versions.
# VersionNumber alone also accepts "1", "v0.1.0" or "01.0.0".
const NUMBER = raw"(?:0|[1-9][0-9]*)"                  # no leading zeros
const PRERELEASE_PART = "(?:$NUMBER|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
const SEMVER = Regex("\\A$NUMBER\\.$NUMBER\\.$NUMBER(?:-$PRERELEASE_PART(?:\\.$PRERELEASE_PART)*)?\\z")

"""
    Model(dir = the fisp600-rank16 artifact)

A NeuralBloch model bundle: `dir/model_card.toml` and the files that it
names, which are an ONNX model for each target and the RF tables. If you do
not give `dir`, `Model` uses the `fisp600-rank16` bundle that the
Artifacts.toml of this package binds. Pkg downloads this bundle from the
NeuralBloch release when it installs the package. `Model` reads all other
bundles, for example your own export, from their directory.

`Model` checks all of the bundle:

- the schema version of the card (1.x),
- the presence and the type of each field. All numbers must be finite.
- the values: a semantic version, a family name that states the rank of the
  card, a positive ntr and rank, training ranges with 0 < low < high, a
  supported domain constraint and derivative method (with steps only for
  forward differences), and a positive TR, TE, max_state, repetitions and
  undersampling factor,
- the files that the card names: relative to `dir`, and in the bundle,
- the sha256 of each ONNX file (64 hexadecimal digits), against the value
  that the card pins,
- the RF tables: one finite number for each readout.

If a check fails, `Model` throws a [`BundleError`](@ref).

The fields of a `Model` hold the data of the card:

- `family` and `version`: the name of the model. A family is for example
  `fisp600-rank16`. A version is a semantic version, for example
  `v"0.1.0-beta.1"`.
- `ntr`: the readouts for each voxel.
- `rank`: the signal basis coefficients for each voxel.
- `onnx` and `sha256`: one for each target (`signal_and_derivatives`,
  `signal`, `coefficients`).
- `files`: the card and all the files that it names, relative to `dir`.
- `domain`: the training ranges of T1, T2 and B1. `constraint`: the domain
  constraint.
- `derivatives`: the method that calculated the derivatives of the training
  targets, and the steps for forward differences.
- `sequence`: the FISP3D sequence, with the RF tables in degrees.
- `provenance`.
"""
struct Model
    dir::String
    family::String
    version::VersionNumber
    ntr::Int
    rank::Int
    onnx::NamedTuple{TARGETS,NTuple{3,String}}
    sha256::NamedTuple{TARGETS,NTuple{3,String}}
    files::Vector{String}
    domain::@NamedTuple{T1::Range, T2::Range, B1::Range}
    constraint::String
    derivatives::@NamedTuple{method::String, dT1_step::Union{Nothing,Float64}, dT2_step::Union{Nothing,Float64}}
    sequence::@NamedTuple{kind::String, TR::Float64, TE::Float64, TI::Float64, TW::Float64,
                          max_state::Int, repetitions::Int, inversion_prepulse::Bool,
                          wait_spoiling::Bool, undersampling_factor::Int, delta_k_spoil::Float64,
                          rf_angle::Vector{Float64}, rf_phase::Vector{Float64}}
    provenance::Dict{String,String}
end

describe(::Type{Float64}) = "a finite number"
describe(::Type{Int}) = "an integer"
describe(::Type{Bool}) = "true or false"
describe(::Type{String}) = "a string"
describe(::Type{Range}) = "two finite numbers [low, high] with 0 < low < high"
describe(::Type{VersionNumber}) =
    "a semantic version MAJOR.MINOR.PATCH[-PRERELEASE] without build metadata, e.g. \"0.1.0-beta.1\""

isnumber(x) = x isa Real && !(x isa Bool)

# Return card[table][key] as a T, or throw a BundleError that names the field.
# A table name with dots is a nested table: "model.signal" is
# card["model"]["signal"].
function field(card, path, table, key, ::Type{T}) where {T}
    t = card
    for name in split(table, '.')
        t = t isa AbstractDict ? get(t, name, nothing) : nothing
    end
    t isa AbstractDict || throw(BundleError("$path: table [$table] is missing"))
    haskey(t, key) || throw(BundleError("$path: [$table] $key is missing"))
    v = t[key]
    T === Float64 && isnumber(v) && isfinite(v) && return Float64(v)
    T === Int && v isa Integer && !(v isa Bool) && return Int(v)
    T === Bool && v isa Bool && return v
    T === String && v isa AbstractString && return String(v)
    T === VersionNumber && v isa AbstractString && occursin(SEMVER, v) && return VersionNumber(v)
    # A range is of T1, T2 or B1. The network takes the log of T1 and T2, and B1
    # is a positive scale factor, thus 0 < low.
    if T === Range && v isa AbstractVector && length(v) == 2 && all(x -> isnumber(x) && isfinite(x), v) &&
       0 < v[1] < v[2]
        return (Float64(v[1]), Float64(v[2]))
    end
    throw(BundleError("$path: [$table] $key must be $(describe(T)), not $(repr(v))"))
end

# Return the path of a file that the card names. The name is relative to the
# bundle, and the file must be in the bundle. Pkg installs only the bundle,
# thus a file outside it is not on the computer of a user.
function bundlefile(dir, path, table, key, name)
    outside() = throw(BundleError(
        "$path: [$table] $key names $name, which is outside the bundle; names are relative to $dir"))
    (isabspath(name) || first(splitpath(normpath(name))) == "..") && outside()
    file = joinpath(dir, name)
    isfile(file) || throw(BundleError("$path: [$table] $key names $name, which $dir does not have"))
    # A symbolic link in the bundle can also point outside it.
    startswith(realpath(file), joinpath(realpath(dir), "")) || outside()
    return file
end

# Read an RF table: one number on each line, `ntr` numbers in total.
function rftable(file, ntr)
    values = Float64[]
    for (i, line) in enumerate(eachline(file))
        isempty(strip(line)) && continue
        v = tryparse(Float64, line)
        v === nothing && throw(BundleError("$file:$i: not a number: $(repr(line))"))
        isfinite(v) || throw(BundleError("$file:$i: not a finite number: $(repr(line))"))
        push!(values, v)
    end
    length(values) == ntr ||
        throw(BundleError("$file has $(length(values)) values; [model] ntr is $ntr (see $CARD)"))
    return values
end

# Return the rank that a family name states, or nothing if the name does not
# agree with the pattern.
function familyrank(family)
    occursin(FAMILY_PATTERN, family) || return nothing
    ranks = [m[1] for m in (match(r"\Arank([1-9][0-9]*)\z", w) for w in split(family, '-')) if m !== nothing]
    return length(ranks) == 1 ? parse(Int, ranks[1]) : nothing
end

function Model(dir::AbstractString = artifact"neuralbloch_fisp600_rank16")
    dir = abspath(dir)
    isdir(dir) || throw(BundleError("$dir does not exist"))
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
    files = [CARD]                     # each file that the card names, relative to dir
    file(table, key) = (f = bundlefile(dir, path, table, key, need(String, table, key));
                        push!(files, relpath(f, dir)); f)
    positive(T, table, key) = (v = need(T, table, key);
                               v > 0 || throw(BundleError("$path: [$table] $key must be positive, not $v")); v)

    schema = get(card, "schema_version", nothing)
    schema === nothing && throw(BundleError("$path has no schema_version; this NeuralBloch reads schema $SCHEMA_MAJOR.x"))
    parts = schema isa AbstractString ? match(Regex("\\A($NUMBER)\\.$NUMBER\\z"), schema) : nothing
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
        "of them rank<N>, e.g. fisp600-rank16"))
    stated == rank || throw(BundleError(
        "$path: [model] family $(repr(family)) states rank $stated, but [model] rank is $rank"))

    pinned = map(TARGETS) do target
        table = "model.$target"
        sha = need(String, table, "sha256")
        occursin(r"\A[0-9a-f]{64}\z", sha) ||
            throw(BundleError("$path: [$table] sha256 must be 64 hexadecimal digits, not $(repr(sha))"))
        onnxfile = file(table, "onnx")
        actual = bytes2hex(open(SHA.sha256, onnxfile))
        actual == sha || throw(BundleError(
            "$onnxfile has sha256 $actual, but $path pins $sha for $target: it is not the model the card describes"))
        (onnxfile, sha)
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
    # A step is necessary for forward differences, and not permitted for the
    # other method.
    function fdstep(key)
        method == "forward difference" && return positive(Float64, "derivatives", key)
        haskey(card["derivatives"], key) && throw(BundleError(
            "$path: [derivatives] $key is given, but steps apply only to forward differences, not $method"))
        return nothing
    end
    derivatives = (; method, dT1_step = fdstep("dT1_step"), dT2_step = fdstep("dT2_step"))

    rf(key) = rftable(file("sequence", key), ntr)
    # TI, TW and delta_k_spoil can be 0. The other numbers cannot.
    sequence = (kind = need(String, "sequence", "kind"),
                TR = positive(Float64, "sequence", "TR"), TE = positive(Float64, "sequence", "TE"),
                TI = need(Float64, "sequence", "TI"), TW = need(Float64, "sequence", "TW"),
                max_state = positive(Int, "sequence", "max_state"),
                repetitions = positive(Int, "sequence", "repetitions"),
                inversion_prepulse = need(Bool, "sequence", "inversion_prepulse"),
                wait_spoiling = need(Bool, "sequence", "wait_spoiling"),
                undersampling_factor = positive(Int, "sequence", "undersampling_factor"),
                delta_k_spoil = need(Float64, "sequence", "delta_k_spoil"),
                rf_angle = rf("rf_angle"), rf_phase = rf("rf_phase"))
    need(String, "provenance", "upstream")
    provenance = Dict{String,String}(k => string(v) for (k, v) in card["provenance"])
    return Model(dir, family, version, ntr, rank, onnx, sha256, sort!(files), domain, constraint, derivatives,
                 sequence, provenance)
end

Base.show(io::IO, m::Model) =
    print(io, "Model(", repr(m.dir), "; ", m.family, " v", m.version, ", ntr = ", m.ntr, ", rank = ", m.rank, ")")

"""
    contract_problems(current::Model, new::Model) -> Vector{String}

Return the problems that prevent bundle `new` from replacing `current` as a
later version of the same model family. If there are no problems, the vector
is empty.

In a family, callers rely on these properties:

- The readouts, the rank, the sequence (with the RF tables) and the domain
  constraint stay the same.
- The training domain stays the same or becomes larger.
- The version increases in semantic-version order
  (0.1.0-beta.1 < 0.1.0-beta.2 < 0.1.0 < 0.1.1).

Two different families have no shared contract. Thus, if `new` is from a
different family, the function returns no problems.
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
