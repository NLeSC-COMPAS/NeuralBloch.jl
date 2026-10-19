# Bind a model release file as the artifact of this package for its model
# family. The export_bundle script of NeuralBloch writes the release file.
#
#     julia --project dev/bind_bundle.jl TARBALL [ARTIFACTS_TOML]
#
# TARBALL is the release file neuralbloch-model-<family>-v<version>.tar.gz:
# the file that you download from the NeuralBloch release
# model-<family>-v<version>, or, before the release exists, the file in
# outputs/ of NeuralBloch. In that case, attach this exact file to the
# release. The default ARTIFACTS_TOML is the Artifacts.toml of this package.
# The script does not publish. If the depot does not have the contents of the
# file, Pkg installs the file from its local path: it unpacks the file and
# checks its two hashes, as it does after it downloads the release. Thus, the
# test of the entry occurs before the release exists.
#
# Requirements for the file:
#
# - It contains a bundle that this NeuralBloch reads, and only the card and
#   the files that the card names.
# - Its name agrees with the family and the version of its card.
# - If ARTIFACTS_TOML binds an earlier version of the family, the new version
#   keeps the contract of the family (NeuralBloch.contract_problems): the same
#   readouts, rank, sequence and RF tables, a training domain that is the same
#   or larger, and a higher version.
#
# If a requirement is not satisfied, the script does not change
# ARTIFACTS_TOML, lists the problems and exits with code 1. If ARTIFACTS_TOML
# binds this file (the same contents, sha256 and URL), the script changes
# nothing and exits with code 0. If it binds the same contents from a
# different file, for example the same bundle compressed again, the script
# refuses the file: the sha256 of a published release does not change. A
# different family gets a new entry. Incorrect arguments give exit code 2.
#
# The entry is neuralbloch_<family>, with underscores in place of hyphens. The
# entry is not lazy: Pkg downloads it with the package, from the release URL,
# and checks the sha256 of the file and its tree hash. Publish the release with
# this exact file before you push a commit that binds it.

using NeuralBloch
using NeuralBloch: BundleError, contract_problems
using Pkg.Artifacts: artifact_hash, artifact_meta, artifact_path, bind_artifact!, download_artifact,
                     ensure_artifact_installed
using SHA: sha256
using Tar: Tar

# The location of the NeuralBloch releases, and the names that export_bundle
# gives to the release and the file of a family version
# (scripts/export_bundle.py: release_tag, tarball_name).
const RELEASES = "https://github.com/NLeSC-COMPAS/NeuralBloch/releases/download"
release_tag(family, version) = "model-$family-v$version"
tarball_name(family, version) = "neuralbloch-$(release_tag(family, version)).tar.gz"
release_url(family, version) = "$RELEASES/$(release_tag(family, version))/$(tarball_name(family, version))"

# The entry of a family in Artifacts.toml, for example
# neuralbloch_fisp600_rank16.
artifact_name(family) = "neuralbloch_" * replace(family, '-' => '_')

# All the files in `dir`, relative to `dir`.
filesunder(dir) = sort!([relpath(joinpath(root, name), dir) for (root, _, names) in walkdir(dir) for name in names])

# Return the bundle in `dir`, or the BundleError that refuses it.
loaded(dir) = try Model(dir) catch err; err isa BundleError ? err : rethrow() end

# Return the exit code. The test suite calls this function with buffers for
# stdout and stderr.
function main(args; stdout::IO = Base.stdout, stderr::IO = Base.stderr)
    if !(1 ≤ length(args) ≤ 2)
        println(stderr, "usage: julia --project dev/bind_bundle.jl TARBALL [ARTIFACTS_TOML]")
        return 2
    end
    tarball = abspath(args[1])
    toml = abspath(length(args) == 2 ? args[2] : joinpath(pkgdir(NeuralBloch), "Artifacts.toml"))
    refuse(why...) = (println(stderr, "Refusing $tarball: ", why...); 1)
    isfile(tarball) || return refuse("there is no such file")

    # Calculate the two hashes from the file: its sha256, and the git tree hash
    # of its contents. The tree hash is the name of the artifact.
    sha = bytes2hex(open(sha256, tarball))
    tree = try
        Base.SHA1(Tar.tree_hash(`gzip -dc $tarball`))
    catch err
        return refuse("it is not a gzip-compressed tarball: ", sprint(showerror, err))
    end
    # If the depot does not have the tree, Pkg unpacks the file and checks the
    # two hashes, as after a download. Unset JULIA_PKG_IGNORE_HASHES: it lets
    # a wrong tree hash pass.
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
    extra = setdiff(filesunder(dir), new.files)
    isempty(extra) || return refuse("it holds files its card does not name: ", join(extra, ", "))

    entry = artifact_name(new.family)
    url = release_url(new.family, new.version)
    bound = isfile(toml) ? artifact_hash(entry, toml) : nothing
    if bound == tree
        # The same contents can come from a different file. Pkg checks the
        # sha256 of the file that it downloads, thus the entry must name
        # this file.
        downloads = get(artifact_meta(entry, toml), "download", [])
        if any(d -> get(d, "url", nothing) == url && get(d, "sha256", nothing) == sha, downloads)
            println(stdout, "$toml already binds $(new.family) v$(new.version) as $entry.")
            return 0
        end
        return refuse("$toml binds its contents as $entry from a different file: ",
                      join(("$(get(d, "url", "?")) (sha256 $(get(d, "sha256", "?")))" for d in downloads), ", "),
                      ". This file has sha256 $sha. Use the file of the release. To replace the file of a ",
                      "version that is not published, see \"Versions and families\" in the README.")
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
