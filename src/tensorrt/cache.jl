# The on-disk cache of compiled native code and built engines. Each entry is
# a directory named after the digest of its key, holding the products and a
# TOML manifest that records the key.
#
# An entry is built in a temporary sibling directory and published with one
# rename, so an entry directory is always complete. A lock per entry (a
# FileWatching.Pidfile) makes concurrent Julia processes build it only once
# (dev/NOTES.md, S8). Temporary names never equal an entry's name.

const MANIFEST = "manifest.toml"

# Seconds after which a lock whose holder died is removed. A live holder
# refreshes its lock every STALE_AGE/2 (it only ever waits on subprocesses,
# so the refresh timer runs).
const STALE_AGE = 10

"""
    cache_root() -> String

NeuralBloch's cache directory, a scratch space of the package in the first
depot of `DEPOT_PATH`. Deleting it is always safe: entries are rebuilt on
demand.
"""
cache_root() = Scratch.@get_scratch!("cache")

"""
    digest(fields::AbstractDict) -> String

The sha256 (hex) of `fields` in canonical form, TOML with sorted keys: equal
contents give equal digests, whatever the insertion order.
"""
function digest(fields::AbstractDict)
    io = IOBuffer()
    TOML.print(io, fields; sorted = true)
    return bytes2hex(SHA.sha256(take!(io)))
end

"""
    engine_key(; onnx, tensorrt, gpu, capability, settings, csrc) -> Dict

Everything that determines an engine: the ONNX's sha256, TensorRT's version
(all four fields; an engine is tied to the exact library that built it), the
GPU's name and compute capability, the build settings, and the sha256 of the
C++ sources the builder is compiled from.
"""
function engine_key(; onnx::AbstractString, tensorrt::VersionNumber, gpu::AbstractString,
                    capability::VersionNumber, settings::AbstractDict, csrc::AbstractString)
    return Dict{String,Any}(
        "onnx_sha256" => String(onnx),
        "tensorrt" => vstring(tensorrt),
        "gpu" => String(gpu),
        "compute_capability" => "$(capability.major).$(capability.minor)",
        "settings" => Dict{String,Any}(settings),
        "csrc_sha256" => String(csrc),
    )
end

"""
    write_manifest(dir, key, info)

Write `dir/manifest.toml`: the entry's `key` and `info`, free-form facts
about the build (a Dict or `nothing`).
"""
function write_manifest(dir::AbstractString, key::AbstractDict, info)
    manifest = Dict{String,Any}("key" => key, "info" => something(info, Dict{String,Any}()))
    open(io -> TOML.print(io, manifest; sorted = true), joinpath(dir, MANIFEST), "w")
    return nothing
end

read_manifest(dir::AbstractString) = TOML.parsefile(joinpath(dir, MANIFEST))

entrypath(root, key) = joinpath(root, digest(key)[1:16])

"""
    lookup(root, key) -> Union{String,Nothing}

The entry directory for `key` under `root`, or `nothing`. The manifest's
recorded key must match, so neither a digest-prefix collision nor a
hand-edited entry is taken for the product.
"""
function lookup(root::AbstractString, key::AbstractDict)
    dir = entrypath(root, key)
    isfile(joinpath(dir, MANIFEST)) || return nothing
    recorded = try
        read_manifest(dir)["key"]
    catch
        return nothing
    end
    return digest(recorded) == digest(key) ? dir : nothing
end

"""
    withlock(f, path, what)

Call `f()` holding the Pidfile lock `path`. When another process holds it,
say once that it is building `what`, then wait for it.
"""
function withlock(f, path::AbstractString, what::AbstractString)
    lock = Pidfile.trymkpidlock(path; stale_age = STALE_AGE)
    if lock === false
        @info "Another process is building $what; waiting for it to finish." lock = path
        lock = Pidfile.mkpidlock(path; stale_age = STALE_AGE)
    end
    try
        return f()
    finally
        close(lock)
    end
end

"""
    getorbuild(build, root, key; what) -> String

The entry directory for `key` under `root`, built first if there is none:
`build(dir)` writes the products into the empty directory `dir` and returns
facts for the manifest (a Dict, or `nothing`).

The build runs holding the entry's lock, in a temporary sibling directory
that one rename publishes. So an entry is always complete, concurrent
processes build it once (the others wait and reuse it), and a build that
throws leaves nothing behind. `what` names the product in the message shown
while waiting.
"""
function getorbuild(build, root::AbstractString, key::AbstractDict; what::AbstractString)
    found = lookup(root, key)
    found === nothing || return found
    mkpath(root)
    path = entrypath(root, key)
    return withlock(path * ".lock", what) do
        existing = lookup(root, key)                  # built while we waited
        existing === nothing || return existing
        # Holding the lock, nobody else writes this entry: anything here is a
        # leftover (a killed build's temporary, an entry for other fields).
        for f in readdir(root; join = true)
            startswith(basename(f), basename(path) * ".tmp-") && rm(f; recursive = true, force = true)
        end
        rm(path; recursive = true, force = true)
        tmp = mktempdir(root; prefix = basename(path) * ".tmp-", cleanup = false)
        try
            write_manifest(tmp, key, build(tmp))
            Base.Filesystem.rename(tmp, path)
        catch
            rm(tmp; recursive = true, force = true)
            rethrow()
        end
        return path
    end
end
