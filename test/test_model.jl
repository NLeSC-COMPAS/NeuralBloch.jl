# model.jl: the bundle (ONNX per target, card, RF tables) and its checks. Julia only.

using NeuralBloch: Model, BundleError, TARGETS, contract_problems
using SHA: sha256
using Tar: Tar
using TOML: TOML

const BUNDLE = joinpath(pkgdir(NeuralBloch), "models")

# The sha256 the bundled card pins for each target's ONNX model.
const PINNED = (signal_and_derivatives = "b4797fa23fad6e1bd8c289ae5d06e929103d4223c090926c6199c35e25fde1b5",
                signal = "d2f1e74b4ea561eeffd5d16406b42138e51df921d5a6f734869c0e28c3241a88",
                coefficients = "2f76136bdecc847682755d279c9cb78bba241b8ca8f2be5f20e1fe59991d60d7")

# A private copy of the bundle whose card.toml is `edit`(original text).
function edited_bundle(edit, dir)
    b = joinpath(dir, "models")
    cp(BUNDLE, b)
    card = joinpath(b, "card.toml")
    write(card, edit(read(card, String)))
    return b
end

tier(:T0, "model: the bundled card") do
    m = Model()
    @test m.dir == BUNDLE
    @test (m.family, m.version, m.ntr, m.rank) == ("fisp600-rank16", 1, 600, 16)
    @test m.onnx == (signal_and_derivatives = joinpath(BUNDLE, "model.onnx"),
                     signal = joinpath(BUNDLE, "model_signal.onnx"),
                     coefficients = joinpath(BUNDLE, "model_coefficients.onnx"))
    @test m.sha256 == PINNED
    @test keys(m.onnx) == keys(m.sha256) == TARGETS
    @test m.domain == (T1 = (0.1, 6.0), T2 = (0.01, 2.0), B1 = (0.5, 1.5))
    @test m.constraint == "T2 < T1"
    @test m.derivatives == (method = "forward sensitivity", dT1_step = nothing, dT2_step = nothing)
    s = m.sequence
    @test (s.kind, s.TR, s.TE, s.TI, s.TW) == ("FISP3D", 0.00558, 0.002745, 0.01084, 0.0)
    @test (s.max_state, s.repetitions, s.undersampling_factor, s.delta_k_spoil) == (64, 5, 1, 0.0)
    @test s.inversion_prepulse && s.wait_spoiling
    @test length(s.rf_angle) == length(s.rf_phase) == 600
    @test (s.rf_angle[1], s.rf_angle[end]) == (3.960833, parse(Float64, last(readlines(joinpath(BUNDLE, "rf_data", "rf_angle_list.txt")))))
    @test s.rf_phase[1] == -1.9641
    @test m.provenance["upstream"] == "NeuralBloch@d5c288465a8b54b146b5ed595837ccceb6b4756e"
    @test (m.provenance["checkpoint_epoch"], m.provenance["blochsimulators"], m.provenance["precision"]) ==
          ("4970", "0.10.0", "FP32")
    @test occursin(r"^[0-9a-f]{64}$", m.provenance["config_sha256"])
    @test sprint(show, m) == "Model($(repr(BUNDLE)); fisp600-rank16 v1, ntr = 600, rank = 16)"
end

tier(:T0, "model: a forward-difference card states its steps") do
    mktempdir() do dir
        fd(s) = replace(s, "method = \"forward sensitivity\"" =>
                           "method = \"forward difference\"\ndT1_step = 0.02\ndT2_step = 0.0025")
        m = Model(edited_bundle(fd, dir))
        @test m.derivatives == (method = "forward difference", dT1_step = 0.02, dT2_step = 0.0025)
    end
end

tier(:T0, "model: malformed cards") do
    sensitivity = "method = \"forward sensitivity\""
    cases = [  # edit of the card => words the error must contain
        (s -> replace(s, r"format = 2[^\n]*\n" => "")) => "has no format",
        (s -> replace(s, "format = 2" => "format = 1")) => "has card format 1; this NeuralBloch reads format 2",
        (s -> replace(s, "format = 2" => "format = \"2\"")) => "has card format \"2\"",
        (s -> replace(s, r"\[domain\].*?\n\n"s => "")) => "table [domain] is missing",
        (s -> replace(s, r"\nntr = 600[^\n]*" => "")) => "[model] ntr is missing",
        (s -> replace(s, "ntr = 600" => "ntr = \"600\"")) => "[model] ntr must be an integer",
        (s -> replace(s, "ntr = 600" => "ntr = 600.0")) => "[model] ntr must be an integer",
        (s -> replace(s, r"\nrank = 16[^\n]*" => "")) => "[model] rank is missing",
        (s -> replace(s, "rank = 16" => "rank = 0")) => "[model] rank must be positive, not 0",
        (s -> replace(s, "rank = 16" => "rank = 12")) => "states rank 16, but [model] rank is 12",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"FISP600\"")) => "[model] family \"FISP600\" must be",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"fisp600-r16\"")) => "exactly one of them rank<N>",
        (s -> replace(s, "version = 1" => "version = 0")) => "[model] version must be positive, not 0",
        (s -> replace(s, r"\[model\.coefficients\].*?\n\n"s => "")) => "table [model.coefficients] is missing",
        (s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [6.0, 0.1]")) => "[domain] T1 must be two numbers [low, high] with low < high",
        (s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.1]")) => "[domain] T1 must be two numbers",
        (s -> replace(s, "TR = 0.00558" => "TR = \"5.58 ms\"")) => "[sequence] TR must be a number",
        (s -> replace(s, "inversion_prepulse = true" => "inversion_prepulse = \"yes\"")) => "[sequence] inversion_prepulse must be true or false",
        (s -> replace(s, "constraint = \"T2 < T1\"" => "constraint = \"T1 < T2\"")) => "unsupported constraint \"T1 < T2\"",
        (s -> replace(s, sensitivity => "method = \"adjoint\"")) => "[derivatives] method \"adjoint\" is not one of",
        (s -> replace(s, sensitivity => "method = \"forward difference\"")) => "[derivatives] dT1_step is missing",
        (s -> replace(s, sensitivity => "method = \"forward difference\"\ndT1_step = -0.02\ndT2_step = 0.0025")) =>
            "[derivatives] dT1_step must be positive",
        (s -> replace(s, sensitivity => "$sensitivity\ndT2_step = 0.0025")) =>
            "dT2_step is given, but steps apply only to forward differences",
        (s -> replace(s, "sha256 = \"" => "sha256 = \"xyz")) => "[model.signal_and_derivatives] sha256 must be 64 hexadecimal digits",
        (s -> replace(s, "[model]" => "[model")) => "is not valid TOML",
        (s -> replace(s, "rf_data/rf_angle_list.txt" => "rf_data/missing.txt")) => "missing.txt, which",
        (s -> replace(s, "onnx = \"model_signal.onnx\"" => "onnx = \"other.onnx\"")) => "[model.signal] onnx names other.onnx, which",
        (s -> replace(s, r"\nupstream = [^\n]*" => "")) => "[provenance] upstream is missing",
    ]
    for (edit, words) in cases
        mktempdir() do dir
            err = @test_throws BundleError Model(edited_bundle(edit, dir))
            msg = sprint(showerror, err.value)
            @test occursin(words, msg)
            @test occursin("card.toml", msg)                  # which file to fix
        end
    end
end

tier(:T0, "model: RF tables must match ntr") do
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        rf = joinpath(b, "rf_data", "rf_phase_list.txt")
        write(rf, join(readlines(rf)[1:599], '\n'))
        err = @test_throws BundleError Model(b)
        @test occursin("rf_phase_list.txt has 599 values; [model] ntr is 600", sprint(showerror, err.value))
        write(rf, "1.0\nnot a number\n")
        err = @test_throws BundleError Model(b)
        @test occursin("rf_phase_list.txt:2: not a number", sprint(showerror, err.value))
    end
end

tier(:T0, "model: each ONNX must be the one the card pins") do
    bundled = Model()
    for target in TARGETS
        mktempdir() do dir
            b = edited_bundle(identity, dir)
            file = basename(bundled.onnx[target])
            write(joinpath(b, file), read(joinpath(b, file))[1:end-1])
            err = @test_throws BundleError Model(b)
            msg = sprint(showerror, err.value)
            @test occursin("pins $(PINNED[target]) for $target", msg)
            @test occursin(file, msg)
        end
    end
    @test_throws BundleError Model("/nonexistent/bundle")
end

tier(:T0, "model: a family's contract between versions") do
    current = Model()
    # A later bundle: version 2 with `edit` applied to its card, and `files`
    # applied to its directory.
    function later(edit; version = 2, files = identity)
        dir = mktempdir()
        b = edited_bundle(s -> edit(replace(s, "version = 1" => "version = $version")), dir)
        files(b)
        return Model(b)
    end
    @test contract_problems(current, later(identity)) == String[]
    @test contract_problems(current, later(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.05, 7.0]"))) == String[]
    cases = [  # the later bundle => words one of the problems must contain
        later(identity; version = 1) => "version 1 does not follow version 1",
        later(s -> replace(s, "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]")) =>
            "domain B1 [0.5, 1.5] → [0.6, 1.5] does not contain the previous range",
        later(s -> replace(s, "T2 = [0.01, 2.0]" => "T2 = [0.01, 1.0]")) => "domain T2",
        later(s -> replace(s, "TR = 0.00558" => "TR = 0.006")) => "sequence TR 0.00558 → 0.006",
        later(s -> replace(s, "max_state = 64" => "max_state = 32")) => "sequence max_state 64 → 32",
        later(identity; files = b -> (f = joinpath(b, "rf_data", "rf_angle_list.txt");
                                      write(f, replace(read(f, String), r"^[^\n]+"m => "1.0"; count = 1)))) =>
            "the rf_angle table differs",
        later(s -> replace(s, "ntr = 600" => "ntr = 599");
              files = b -> foreach(("rf_angle_list.txt", "rf_phase_list.txt")) do name
                  f = joinpath(b, "rf_data", name)
                  write(f, join(readlines(f)[1:599], '\n'))
              end) => "ntr 600 → 599",
    ]
    for (new, words) in cases
        problems = contract_problems(current, new)
        @test any(problem -> occursin(words, problem), problems)
    end
    # Another family is another contract: nothing to compare
    other = later(s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"fisp600b-rank16\"",
                                  "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]"))
    @test contract_problems(current, other) == String[]
end

# dev/update_bundle.jl's functions, for its main in this process: a Julia
# process per call would take seconds to start.
module UpdateBundle
include(joinpath(@__DIR__, "..", "dev", "update_bundle.jl"))
end

tier(:T0, "model: update_bundle.jl installs within the family's contract") do
    function update(src, dest)
        out, err = IOBuffer(), IOBuffer()
        code = UpdateBundle.main([src, dest]; stdout = out, stderr = err)
        return code == 0, String(take!(out)), String(take!(err))
    end
    mktempdir() do dir
        dest = edited_bundle(identity, mkpath(joinpath(dir, "dest")))           # v1, as bundled
        v2 = edited_bundle(s -> replace(s, "version = 1" => "version = 2", "T1 = [0.1, 6.0]" => "T1 = [0.1, 7.0]"),
                           mkpath(joinpath(dir, "v2")))
        write(joinpath(v2, "stray.txt"), "not named by the card")
        # Once as the command the README gives (read throws unless it exits with 0)
        script = joinpath(pkgdir(NeuralBloch), "dev", "update_bundle.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $v2 $dest`
        @test occursin("Installed fisp600-rank16 v2", read(cmd, String))
        @test (Model(dest).version, Model(dest).domain.T1) == (2, (0.1, 7.0))
        @test !isfile(joinpath(dest, "stray.txt"))                             # only what the card names
        @test readdir(dirname(dest)) == ["models"]                             # no temporaries left

        ok, out, _ = update(v2, dest)
        @test ok && occursin("already holds fisp600-rank16 v2", out)

        narrower = edited_bundle(s -> replace(s, "version = 1" => "version = 3", "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]"),
                                 mkpath(joinpath(dir, "narrower")))
        before = read(joinpath(dest, "card.toml"))
        ok, _, err = update(narrower, dest)
        @test !ok && occursin("domain B1 [0.5, 1.5] → [0.6, 1.5]", err)
        @test read(joinpath(dest, "card.toml")) == before                      # unchanged

        broken = edited_bundle(s -> replace(s, "format = 2" => "format = 3"), mkpath(joinpath(dir, "broken")))
        ok, _, err = update(broken, dest)
        @test !ok && occursin("Refusing", err) && occursin("format 3", err)
        @test read(joinpath(dest, "card.toml")) == before
    end
end

# dev/bind_bundle.jl's functions, for its main in this process.
module BindBundle
include(joinpath(@__DIR__, "..", "dev", "bind_bundle.jl"))
end

# The gzip-compressed tarball of `dir` at `path`, its files at the root, as
# NeuralBloch's export_bundle writes a release file.
function release_file(dir, path)
    open(pipeline(`gzip -9n`; stdout = path), "w") do io
        Tar.create(dir, io)
    end
    return path
end

tier(:T0, "model: bind_bundle.jl binds a release file within the family's contract") do
    mktempdir() do dir
        depot = mkpath(joinpath(dir, "depot"))           # artifacts land here, not in ~/.julia
        toml = joinpath(dir, "Artifacts.toml")
        entry() = TOML.parsefile(toml)["neuralbloch_fisp600_rank16"]
        url(version) = "https://github.com/NLeSC-COMPAS/NeuralBloch/releases/download/" *
                       "model-fisp600-rank16-v$version/neuralbloch-model-fisp600-rank16-v$version.tar.gz"
        # Version `version` of the bundle as a release file named `name`, with
        # `edit` applied to its card and `files` to its directory
        function release(edit, version; files = identity, name = basename(url(version)))
            b = edited_bundle(s -> edit(replace(s, "version = 1" => "version = $version")),
                              mkpath(joinpath(dir, "v$version")))
            files(b)
            return release_file(b, joinpath(dir, name))
        end
        function bind(file)
            out, err = IOBuffer(), IOBuffer()
            pushfirst!(DEPOT_PATH, depot)
            code = try
                BindBundle.main([file, toml]; stdout = out, stderr = err)
            finally
                popfirst!(DEPOT_PATH)
            end
            return code == 0, String(take!(out)), String(take!(err))
        end

        v1 = release(identity, 1)
        ok, out, _ = bind(v1)
        @test ok && occursin("Bound fisp600-rank16 v1 as neuralbloch_fisp600_rank16", out)
        @test entry()["git-tree-sha1"] == Tar.tree_hash(`gzip -dc $v1`)
        @test entry()["download"] == [Dict("sha256" => bytes2hex(open(sha256, v1)), "url" => url(1))]
        @test !haskey(entry(), "lazy")                                       # downloaded with the package
        ok, out, _ = bind(v1)
        @test ok && occursin("already binds fisp600-rank16 v1", out)

        v2 = release(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.1, 7.0]"), 2)
        ok, _, _ = bind(v2)
        @test ok && entry()["download"][1]["url"] == url(2)
        before = read(toml)
        truncated = joinpath(dir, "truncated.tar.gz")
        write(truncated, read(v2)[1:1000])
        cases = [  # a release file to refuse => words the refusal must contain
            release(s -> replace(s, "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]"), 3) =>
                "domain B1 [0.5, 1.5] → [0.6, 1.5]",
            release(identity, 4; files = b -> write(joinpath(b, "stray.txt"), "not named by the card")) =>
                "files its card does not name: stray.txt",
            release(identity, 5; name = basename(url(6))) => "must be named $(basename(url(5)))",
            release(s -> replace(s, "format = 2" => "format = 9"), 7) => "has card format 9",
            truncated => "not a gzip-compressed tarball",
        ]
        for (file, words) in cases
            ok, _, err = bind(file)
            @test !ok && occursin(words, err)
            @test read(toml) == before                                      # unchanged
        end

        # What is bound is what Pkg downloads: v2's entry, its URL swapped for
        # the local file, installs into an empty depot, with neither the
        # package server nor the network
        offline = joinpath(dir, "Offline.toml")
        write(offline, replace(read(toml, String), url(2) => "file://$v2"))
        fresh = mkpath(joinpath(dir, "fresh"))
        code = "using Pkg.Artifacts; print(ensure_artifact_installed(\"neuralbloch_fisp600_rank16\", $(repr(offline))))"
        env = merge(ENV, Dict("JULIA_DEPOT_PATH" => fresh * ":", "JULIA_PKG_SERVER" => ""))
        delete!(env, "JULIA_LOAD_PATH")
        @test startswith(readchomp(setenv(`$(Base.julia_cmd()) --startup-file=no -e $code`, env)), fresh)

        # Once as the command the README gives (read throws unless it exits with 0)
        script = joinpath(pkgdir(NeuralBloch), "dev", "bind_bundle.jl")
        v8 = release(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.1, 8.0]"), 8)
        env = merge(ENV, Dict("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], ':')))
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $v8 $toml`
        @test occursin("Bound fisp600-rank16 v8", read(setenv(cmd, env), String))
        @test entry()["download"][1]["url"] == url(8)
    end
end
