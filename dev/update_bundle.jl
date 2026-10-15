# Install a model bundle that NeuralBloch's export_bundle wrote, into this
# package's models/ or another bundle directory:
#
#     julia --project dev/update_bundle.jl SRC [DEST]
#
# SRC is the bundle (NeuralBloch's outputs/bundle); DEST defaults to models/.
# SRC is loaded first, so a malformed or damaged bundle is refused. When DEST
# holds an earlier version of the same family, SRC must keep the family's
# contract (NeuralBloch.contract_problems): the same readouts, rank, sequence
# and RF tables, a training domain that is the same or wider, and a higher
# version. Otherwise nothing changes and the problems are listed (exit code
# 1). Another family, or a card from before format 2, is replaced with a note.
# An identical bundle is already installed (exit code 0).
#
# Exactly the card and the files it names are copied, into a directory next
# to DEST that then takes its place, so DEST never holds a mix of two bundles.

using NeuralBloch
using NeuralBloch: BundleError, contract_problems
using TOML: TOML

# The card and every file it names, relative to the bundle.
function bundlefiles(dir)
    card = TOML.parsefile(joinpath(dir, "card.toml"))
    onnx = [card["model"][String(target)]["onnx"] for target in NeuralBloch.TARGETS]
    return ["card.toml"; onnx; card["sequence"]["rf_angle"]; card["sequence"]["rf_phase"]]
end

function install(src, dest)
    parent = dirname(dest)
    new = mktempdir(parent; prefix = ".$(basename(dest))-new-", cleanup = false)
    old = joinpath(parent, ".$(basename(dest))-old")
    moved = false
    try
        for name in bundlefiles(src)
            mkpath(dirname(joinpath(new, name)))
            cp(joinpath(src, name), joinpath(new, name))
        end
        chmod(new, 0o755)
        if ispath(dest)
            rm(old; recursive = true, force = true)
            mv(dest, old)
            moved = true
        end
        mv(new, dest)
    catch
        rm(new; recursive = true, force = true)
        moved && !ispath(dest) && mv(old, dest)
        rethrow()
    end
    moved && rm(old; recursive = true)
    return dest
end

function main(args)
    if !(1 ≤ length(args) ≤ 2)
        println(stderr, "usage: julia --project dev/update_bundle.jl SRC [DEST]")
        return 2
    end
    src = abspath(args[1])
    dest = abspath(length(args) == 2 ? args[2] : joinpath(pkgdir(NeuralBloch), "models"))
    new = try
        Model(src)
    catch err
        err isa BundleError || rethrow()
        println(stderr, "Refusing $src: ", sprint(showerror, err))
        return 1
    end
    current = try
        Model(dest)
    catch err
        err isa BundleError || rethrow()
        println("$dest holds no bundle this NeuralBloch reads (", sprint(showerror, err), "); replacing it.")
        nothing
    end
    if current !== nothing
        if read(joinpath(src, "card.toml")) == read(joinpath(dest, "card.toml")) && current.sequence == new.sequence
            println("$dest already holds $(new.family) v$(new.version).")
            return 0
        end
        problems = contract_problems(current, new)
        if !isempty(problems)
            println(stderr, "Refusing to replace $(current.family) v$(current.version) in $dest with v$(new.version), ",
                    "which breaks the family's contract:")
            foreach(problem -> println(stderr, "  - ", problem), problems)
            return 1
        end
        current.family == new.family ||
            println("Replacing family $(current.family) with $(new.family): a different contract for callers.")
    end
    install(src, dest)
    installed = Model(dest)
    println("Installed $(installed.family) v$(installed.version) into $dest.")
    return 0
end

exit(main(ARGS))
