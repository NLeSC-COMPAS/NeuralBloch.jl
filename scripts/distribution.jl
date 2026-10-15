# Distribution test (task 11): does NeuralBloch.jl work installed the way a
# user installs it? A report, not a unit test: it downloads the registry and
# every dependency into an empty depot, which takes minutes and network.
#
#     julia --project=scripts scripts/distribution.jl
#
# Conditions (plan.md): a fresh depot (a temporary JULIA_DEPOT_PATH);
# installed with Pkg.add(url = …), i.e. from this repository's committed
# HEAD, so neither uncommitted files nor sibling repositories can help; a
# read-only package directory. Done when Engine() compiles and builds its
# engines by itself and every target's predictions are correct: equal,
# within the tolerance between two builds of the same ONNX (dev/NOTES.md,
# D18), to what the development setup predicts, which scripts/validate.jl
# checks against EPG.
#
# The model bundle is the artifact Artifacts.toml binds: Pkg.add downloads it
# from NeuralBloch's release into the fresh depot and checks both its hashes.
# Until that release exists, the install fails; run this after publishing it.
#
# Everything lands in a temporary directory under tempdir(), removed at the
# end. The exit code is 1 if a check fails.

using NeuralBloch
using NeuralBloch: TARGETS
using CUDACore: CuArray, synchronize
using Serialization: serialize

const REPO = pkgdir(NeuralBloch)
const TOLERANCE = 5e-6                        # relative, between builds (D18)

# Deterministic voxels inside the training domain.
function voxels(N)
    u(k, j) = mod(0.1k + j * (0.618034, 0.414214, 0.732051)[k], 1)
    T₁ = [exp(log(0.1) + u(1, j) * log(50)) for j in 1:N]
    T₂ = [min(exp(log(0.01) + u(2, j) * log(200)), 0.8T₁[j]) for j in 1:N]
    B₁ = [0.8 + 0.4u(3, j) for j in 1:N]
    return map(x -> Float32.(x), (; T₁, T₂, B₁))
end

# What the child runs, in the fresh depot. It prints one "check: <name> <ok>"
# line per check; everything else it prints is the first-use experience.
const CHILD = raw"""
using Pkg, Serialization
repo, reference = ARGS
Pkg.add(url = repo)
Pkg.add("CUDACore")
using NeuralBloch, CUDACore

check(name, ok) = println("check: ", name, " ", ok)
pkg = pkgdir(NeuralBloch)
check("installed from the depot", startswith(pkg, first(DEPOT_PATH)))
check("bundle downloaded into the depot", startswith(Model().dir, first(DEPOT_PATH)))
check("no git repositories next to the package", !any(d -> isdir(joinpath(pkg, "..", d, ".git")), readdir(dirname(pkg))))
# Read-only, whatever Pkg did: nothing may ever be written there.
run(`chmod -R a-w $pkg`)
snapshot() = [(f, mtime(f)) for (root, _, files) in walkdir(pkg) for f in joinpath.(root, files)]
before = snapshot()

NeuralBloch.versioninfo()
check("functional", NeuralBloch.functional())
engine = Engine()                               # first use: compile, build, cache
println(engine)
v, expected = deserialize(reference)
for (target, ref) in pairs(expected)              # every target: every ONNX model was installed
    out = predict(engine, map(CuArray, v); target)
    synchronize()
    worst = maximum(k -> maximum(abs, Array(out[k]) .- ref[k]) / maximum(abs, ref[k]), keys(ref))
    println("largest relative difference from the development build, $target: ", worst)
    check("$target predictions match the development build", worst <= parse(Float64, ARGS[3]))
end
check("cache in the fresh depot", startswith(NeuralBloch.TensorRT.cache_root(), first(DEPOT_PATH)))
check("package directory untouched", snapshot() == before)
close(engine)
second = @elapsed (e2 = Engine(); close(e2))
println("second Engine(): ", round(second; digits = 2), " s (from the cache)")
run(`chmod -R u+w $pkg`)                        # so the temporary depot can be removed
"""

function main()
    status = readchomp(`git -C $REPO status --porcelain`)
    isempty(status) || @warn "The repository has uncommitted changes; Pkg.add installs the committed HEAD" status
    root = mktempdir()
    reference = joinpath(root, "reference.jls")
    println("Reference outputs from the development setup ($(REPO))")
    v = voxels(3000)
    engine = Engine()
    expected = NamedTuple{TARGETS}(map(target -> map(Array, predict(engine, map(CuArray, v); target)), TARGETS))
    close(engine)
    serialize(reference, (v, expected))

    depot = joinpath(root, "depot")
    env = merge(ENV, Dict("JULIA_DEPOT_PATH" => depot * ":", "JULIA_PROJECT" => joinpath(root, "env"),
                          "JULIA_PKG_OFFLINE" => "false"))
    delete!(env, "JULIA_LOAD_PATH")
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e $CHILD $REPO $reference $TOLERANCE`, env)
    println("Fresh depot: $depot\n", "="^72)
    output = IOBuffer()
    t = @elapsed ok = success(pipeline(cmd; stdout = output, stderr = output))
    text = String(take!(output))
    println(text, "="^72)
    checks = [(m[1], m[2] == "true") for m in eachmatch(r"^check: (.+) (true|false)$"m, text)]
    for (name, passed) in checks
        println(passed ? "PASS  " : "FAIL  ", name)
    end
    println("first use in the fresh depot: ", round(t / 60; digits = 1), " minutes, downloads included")
    good = ok && length(checks) == 6 + length(TARGETS) && all(last, checks)
    println(good ? "Distribution test passed." : "Distribution test FAILED.")
    rm(root; recursive = true, force = true)
    return good
end

exit(main() ? 0 : 1)
