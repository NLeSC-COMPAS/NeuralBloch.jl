# Bind a model release file, written by NeuralBloch's export_bundle, as this
# package's artifact for its model family:
#
#     julia --project dev/bind_bundle.jl TARBALL [ARTIFACTS_TOML]
#
# TARBALL is NeuralBloch's outputs/neuralbloch-model-<family>-v<version>.tar.gz,
# the exact file to attach to NeuralBloch's release model-<family>-v<version>;
# ARTIFACTS_TOML defaults to this package's Artifacts.toml. Nothing is
# downloaded or published: Pkg installs the file from its local path,
# unpacking it and checking both its hashes as it will after downloading the
# release, so the entry is tested before the release exists.
#
# The file must hold a bundle this NeuralBloch reads, nothing but the card and
# the files it names, under the name its card's family and version give. When
# ARTIFACTS_TOML binds an earlier version of the family, the new one must keep
# the family's contract (NeuralBloch.contract_problems): the same readouts,
# rank, sequence and RF tables, a training domain that is the same or wider,
# and a higher version. Otherwise nothing changes and the problems are listed
# (exit code 1). A file that is already bound changes nothing (exit code 0);
# another family gets an entry of its own.
#
# The entry, neuralbloch_<family> with hyphens as underscores, is not lazy:
# Pkg downloads it with the package, from the release URL, and checks the
# file's sha256 and its tree hash. Publish the release with this exact file
# before pushing a commit that binds it.

using NeuralBloch
using NeuralBloch: BundleError, contract_problems
using Pkg.Artifacts: artifact_hash, artifact_path, bind_artifact!, download_artifact, ensure_artifact_installed
using SHA: sha256
using Tar: Tar
using TOML: TOML

# Where NeuralBloch publishes its releases, and the names its export_bundle
# gives a family version's release and file (scripts/export_bundle.py:
# release_tag, tarball_name).
const RELEASES = "https://github.com/NLeSC-COMPAS/NeuralBloch/releases/download"
release_tag(family, version) = "model-$family-v$version"
tarball_name(family, version) = "neuralbloch-$(release_tag(family, version)).tar.gz"
release_url(family, version) = "$RELEASES/$(release_tag(family, version))/$(tarball_name(family, version))"

# A family's entry in Artifacts.toml, e.g. neuralbloch_fisp600_rank16.
artifact_name(family) = "neuralbloch_" * replace(family, '-' => '_')

# The card and every file it names, relative to the bundle.
function bundlefiles(dir)
    card = TOML.parsefile(joinpath(dir, NeuralBloch.CARD))
    onnx = [card["model"][String(target)]["onnx"] for target in NeuralBloch.TARGETS]
    return sort!([NeuralBloch.CARD; onnx; card["sequence"]["rf_angle"]; card["sequence"]["rf_phase"]])
end

# Every file under `dir`, relative to it.
filesunder(dir) = sort!([relpath(joinpath(root, name), dir) for (root, _, names) in walkdir(dir) for name in names])

# The bundle in `dir`, or the BundleError that refuses it.
loaded(dir) = try Model(dir) catch err; err isa BundleError ? err : rethrow() end

# The exit code; the test suite calls it with buffers for stdout and stderr.
function main(args; stdout::IO = Base.stdout, stderr::IO = Base.stderr)
    if !(1 ≤ length(args) ≤ 2)
        println(stderr, "usage: julia --project dev/bind_bundle.jl TARBALL [ARTIFACTS_TOML]")
        return 2
    end
    tarball = abspath(args[1])
    toml = abspath(length(args) == 2 ? args[2] : joinpath(pkgdir(NeuralBloch), "Artifacts.toml"))
    refuse(why...) = (println(stderr, "Refusing $tarball: ", why...); 1)
    isfile(tarball) || return refuse("there is no such file")

    # Both hashes from the file itself: its sha256, and the git tree hash of
    # what it holds, which names the artifact
    sha = bytes2hex(open(sha256, tarball))
    tree = try
        Base.SHA1(Tar.tree_hash(`gzip -dc $tarball`))
    catch err
        return refuse("it is not a gzip-compressed tarball: ", sprint(showerror, err))
    end
    # Pkg's own unpacking and checks of both hashes, as after a download.
    # JULIA_PKG_IGNORE_HASHES would let a tree hash mismatch pass.
    installed = withenv("JULIA_PKG_IGNORE_HASHES" => nothing) do
        download_artifact(tree, "file://" * tarball, sha; quiet_download = true, io = stderr)
    end
    installed === true || return refuse("Pkg cannot install it: ", sprint(showerror, installed))
    dir = artifact_path(tree)

    new = loaded(dir)
    new isa BundleError && return refuse(sprint(showerror, new))
    name = tarball_name(new.family, new.version)
    basename(tarball) == name ||
        return refuse("its card is $(new.family) v$(new.version), so the release file must be named $name")
    extra = setdiff(filesunder(dir), bundlefiles(dir))
    isempty(extra) || return refuse("it holds files its card does not name: ", join(extra, ", "))

    entry = artifact_name(new.family)
    bound = isfile(toml) ? artifact_hash(entry, toml) : nothing
    if bound == tree
        println(stdout, "$toml already binds $(new.family) v$(new.version) as $entry.")
        return 0
    elseif bound === nothing
        println(stdout, "Adding family $(new.family) to $toml as $entry.")
    else
        current = loaded(ensure_artifact_installed(entry, toml; io = stderr))
        if current isa BundleError
            println(stdout, "$toml binds $entry to a bundle this NeuralBloch does not read (",
                    sprint(showerror, current), "); replacing it.")
        else
            problems = contract_problems(current, new)
            if !isempty(problems)
                println(stderr, "Refusing to replace $(current.family) v$(current.version) in $toml with ",
                        "v$(new.version), which breaks the family's contract:")
                foreach(problem -> println(stderr, "  - ", problem), problems)
                return 1
            end
        end
    end
    url = release_url(new.family, new.version)
    bind_artifact!(toml, entry, tree; download_info = [(url, sha)], lazy = false, force = true)
    println(stdout, "Bound $(new.family) v$(new.version) as $entry in $toml:\n",
            "  git-tree-sha1 = $(bytes2hex(tree.bytes))\n  sha256 = $sha\n  url = $url\n",
            "Publish $tarball unchanged as that URL, in NeuralBloch's release ",
            "$(release_tag(new.family, new.version)), before pushing a commit that binds it.")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
