# model.jl: the bundle (ONNX per target, card, RF tables) and its checks. Julia only.

using NeuralBloch: Model, BundleError, TARGETS, contract_problems
using Pkg.Artifacts: artifact_hash, artifact_path
using SHA: sha256
using Tar: Tar
using TOML: TOML

# The sha256 the default card pins for each target's ONNX model.
const PINNED = (signal_and_derivatives = "757a2fd6a70e4ac6aad3b89596ea0c6f63f17d8353c97169cc43db9c86947cf4",
                signal = "0e23650b9df0b2680c40c2a7f13a6fd1e607195a3f844baca0920de7f9ea8c25",
                coefficients = "c5e5b642208c941b87896791853b5e423f3a842fb6bd38a8cb6c1edcb1d4f0bc")

tier(:T0, "model: the default bundle") do
    m = Model()
    # The artifact that Artifacts.toml binds
    @test m.dir == artifact_path(artifact_hash("neuralbloch_fisp600_rank16", joinpath(pkgdir(NeuralBloch), "Artifacts.toml")))
    @test (m.family, m.version, m.ntr, m.rank) == ("fisp600-rank16", v"0.1.0-beta.1", 600, 16)
    @test m.onnx == (signal_and_derivatives = joinpath(BUNDLE, "model.onnx"),
                     signal = joinpath(BUNDLE, "model_signal.onnx"),
                     coefficients = joinpath(BUNDLE, "model_coefficients.onnx"))
    @test m.sha256 == PINNED
    @test keys(m.onnx) == keys(m.sha256) == TARGETS
    @test m.files == ["model.onnx", "model_card.toml", "model_coefficients.onnx", "model_signal.onnx",
                      "rf_data/rf_angle_list.txt", "rf_data/rf_phase_list.txt"]
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
    @test m.provenance["upstream"] == "NeuralBloch@0f18445ba5f6453c1f4d2f59d8da7e4b0afde06d"
    @test (m.provenance["checkpoint_epoch"], m.provenance["blochsimulators"], m.provenance["precision"]) ==
          ("4970", "0.10.0", "FP32")
    @test occursin(r"^[0-9a-f]{64}$", m.provenance["config_sha256"])
    @test sprint(show, m) == "Model($(repr(BUNDLE)); fisp600-rank16 v0.1.0-beta.1, ntr = 600, rank = 16)"
end

tier(:T0, "model: a forward-difference card states its steps") do
    mktempdir() do dir
        fd(s) = replace(s, "method = \"forward sensitivity\"" =>
                           "method = \"forward difference\"\ndT1_step = 0.02\ndT2_step = 0.0025")
        m = Model(edited_bundle(fd, dir))
        @test m.derivatives == (method = "forward difference", dT1_step = 0.02, dT2_step = 0.0025)
    end
end

tier(:T0, "model: a newer MINOR of the card's schema only adds fields") do
    mktempdir() do dir
        newer(s) = replace(s, "schema_version = \"1.0\"" => "schema_version = \"1.3\"",
                           "rank = 16" => "rank = 16\nnote = \"a field of schema 1.3\"") * "\n[added]\nkey = 1\n"
        m, default = Model(edited_bundle(newer, dir)), Model()
        @test (m.family, m.version, m.ntr, m.rank, m.domain, m.sequence) ==
              (default.family, default.version, default.ntr, default.rank, default.domain, default.sequence)
    end
    # A card from before the schema: card.toml, without model_card.toml
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        mv(joinpath(b, "model_card.toml"), joinpath(b, "card.toml"))
        err = @test_throws BundleError Model(b)
        @test occursin("has card.toml, a card from before schema version 1.0, but no model_card.toml",
                       sprint(showerror, err.value))
    end
end

tier(:T0, "model: malformed cards") do
    sensitivity = "method = \"forward sensitivity\""
    version = "version = \"0.1.0-beta.1\""
    semantic = "[model] version must be a semantic version MAJOR.MINOR.PATCH[-PRERELEASE] without build metadata"
    schema = "schema_version = \"1.0\""
    cases = [  # edit of the card => words the error must contain
        (s -> replace(s, r"schema_version = [^\n]*\n" => "")) => "has no schema_version; this NeuralBloch reads schema 1.x",
        (s -> replace(s, schema => "schema_version = \"2.0\"")) => "has schema_version 2.0; this NeuralBloch reads schema 1.x",
        (s -> replace(s, schema => "schema_version = \"0.9\"")) => "has schema_version 0.9; this NeuralBloch reads schema 1.x",
        (s -> replace(s, schema => "schema_version = 1.0")) => "schema_version must be \"MAJOR.MINOR\", e.g. \"1.0\", not 1.0",
        (s -> replace(s, schema => "schema_version = \"1\"")) => "schema_version must be \"MAJOR.MINOR\"",
        (s -> replace(s, schema => "schema_version = \"1.0.0\"")) => "schema_version must be \"MAJOR.MINOR\"",
        (s -> replace(s, schema => "schema_version = \"01.0\"")) => "schema_version must be \"MAJOR.MINOR\"",
        (s -> replace(s, r"\[domain\].*?\n\n"s => "")) => "table [domain] is missing",
        (s -> replace(s, r"\nntr = 600[^\n]*" => "")) => "[model] ntr is missing",
        (s -> replace(s, "ntr = 600" => "ntr = \"600\"")) => "[model] ntr must be an integer",
        (s -> replace(s, "ntr = 600" => "ntr = 600.0")) => "[model] ntr must be an integer",
        (s -> replace(s, r"\nrank = 16[^\n]*" => "")) => "[model] rank is missing",
        (s -> replace(s, "rank = 16" => "rank = 0")) => "[model] rank must be positive, not 0",
        (s -> replace(s, "rank = 16" => "rank = 12")) => "states rank 16, but [model] rank is 12",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"FISP600\"")) => "[model] family \"FISP600\" must be",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"fisp600-r16\"")) => "exactly one of them rank<N>",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"fisp600-rank16-rank8\"")) => "exactly one of them rank<N>",
        (s -> replace(s, "family = \"fisp600-rank16\"" => "family = 1")) => "[model] family must be a string, not 1",
        (s -> replace(s, r"\nversion = [^\n]*" => "")) => "[model] version is missing",
        (s -> replace(s, version => "version = 1")) => "$semantic, e.g. \"0.1.0-beta.1\", not 1",
        (s -> replace(s, version => "version = \"v0.1.0\"")) => semantic,
        (s -> replace(s, version => "version = \"01.0.0\"")) => semantic,
        (s -> replace(s, version => "version = \"0.1.0-beta.01\"")) => semantic,
        (s -> replace(s, version => "version = \"0.1.0+build.1\"")) => semantic,
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
        # A file outside the bundle, also one that exists, is not in the release
        (s -> replace(s, "\"rf_data/rf_angle_list.txt\"" => repr(joinpath(BUNDLE, "rf_data", "rf_angle_list.txt")))) =>
            "[sequence] rf_angle names $(joinpath(BUNDLE, "rf_data", "rf_angle_list.txt")), which is outside the bundle",
        (s -> replace(s, "\"rf_data/rf_phase_list.txt\"" => "\"rf_data/../../rf_phase_list.txt\"")) =>
            "[sequence] rf_phase names rf_data/../../rf_phase_list.txt, which is outside the bundle",
        (s -> replace(s, r"\nupstream = [^\n]*" => "")) => "[provenance] upstream is missing",
    ]
    for (edit, words) in cases
        mktempdir() do dir
            err = @test_throws BundleError Model(edited_bundle(edit, dir))
            msg = sprint(showerror, err.value)
            @test occursin(words, msg)
            @test occursin("model_card.toml", msg)            # which file to fix
        end
    end
end

tier(:T0, "model: RF tables must match ntr") do
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        rf = joinpath(b, "rf_data", "rf_phase_list.txt")
        values = readlines(rf)
        write(rf, "\n" * join(values, "\n\n") * "\n \n")          # blank lines do not count
        @test Model(b).sequence.rf_phase == Model().sequence.rf_phase
        write(rf, join(values[1:599], '\n'))
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
    err = @test_throws BundleError Model("/nonexistent/bundle")
    @test occursin("/nonexistent/bundle does not exist", sprint(showerror, err.value))
end

tier(:T0, "model: a symbolic link must not point outside the bundle") do
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        rf = joinpath(b, "rf_data", "rf_angle_list.txt")
        # A link to a file in the bundle is a file of the bundle
        mv(rf, joinpath(b, "rf_data", "angles.txt"))
        symlink("angles.txt", rf)
        @test Model(b).sequence.rf_angle == Model().sequence.rf_angle
        # A link to a file outside the bundle is not
        outside = joinpath(dir, "angles.txt")
        mv(joinpath(b, "rf_data", "angles.txt"), outside)
        rm(rf)
        symlink(outside, rf)
        err = @test_throws BundleError Model(b)
        @test occursin("[sequence] rf_angle names rf_data/rf_angle_list.txt, which is outside the bundle",
                       sprint(showerror, err.value))
    end
end

tier(:T0, "model: a relative directory") do
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        m = cd(() -> Model("bundle"), dir)
        @test isabspath(m.dir) && samefile(m.dir, b)
        @test all(f -> isabspath(f) && isfile(f), m.onnx)
    end
end

tier(:T0, "model: a family's contract between versions") do
    current = Model()
    # A later bundle (see edited_bundle).
    later(edit; version = "0.1.0-beta.2", files = identity) = Model(edited_bundle(edit, mktempdir(); version, files))
    @test contract_problems(current, later(identity)) == String[]
    @test contract_problems(current, later(identity; version = "0.1.0")) == String[]   # the release after its betas
    @test contract_problems(current, later(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.05, 7.0]"))) == String[]
    cases = [  # the later bundle => words that one of the problems must contain
        later(identity; version = "0.1.0-beta.1") => "version 0.1.0-beta.1 does not follow version 0.1.0-beta.1",
        # semantic-version order: alpha before beta, a pre-release before its release
        later(identity; version = "0.1.0-alpha.9") => "version 0.1.0-alpha.9 does not follow version 0.1.0-beta.1",
        later(identity; version = "0.0.9") => "version 0.0.9 does not follow version 0.1.0-beta.1",
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
    # A different family has a different contract: there is nothing to compare
    other = later(s -> replace(s, "family = \"fisp600-rank16\"" => "family = \"fisp600b-rank16\"",
                                  "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]"))
    @test contract_problems(current, other) == String[]
end

# The functions of dev/bind_bundle.jl, to run its main in this process. A Julia
# process for each call takes seconds to start.
module BindBundle
include(joinpath(@__DIR__, "..", "dev", "bind_bundle.jl"))
end

# Write the gzip-compressed tarball of `dir` to `path`, with the files at the
# root, as the export_bundle script of NeuralBloch writes a release file.
function release_file(dir, path)
    open(pipeline(`gzip -9n`; stdout = path), "w") do io
        Tar.create(dir, io)
    end
    return path
end

tier(:T0, "model: bind_bundle.jl binds a release file within the family's contract") do
    mktempdir() do dir
        depot = mkpath(joinpath(dir, "depot"))           # the artifacts go here, not into ~/.julia
        toml = joinpath(dir, "Artifacts.toml")
        entry() = TOML.parsefile(toml)["neuralbloch_fisp600_rank16"]
        url(version) = "https://github.com/NLeSC-COMPAS/NeuralBloch/releases/download/" *
                       "model-fisp600-rank16-v$version/neuralbloch-model-fisp600-rank16-v$version.tar.gz"
        # Version `version` of the bundle (see edited_bundle) as a release file
        # with the name `name`.
        release(edit, version; files = identity, name = basename(url(version))) =
            release_file(edited_bundle(edit, mkpath(joinpath(dir, "v$version")); version, files), joinpath(dir, name))
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

        v1 = release(identity, "0.1.0-beta.1")
        ok, out, _ = bind(v1)
        @test ok && occursin("Bound fisp600-rank16 v0.1.0-beta.1 as neuralbloch_fisp600_rank16", out)
        @test entry()["git-tree-sha1"] == Tar.tree_hash(`gzip -dc $v1`)
        @test entry()["download"] == [Dict("sha256" => bytes2hex(open(sha256, v1)), "url" => url("0.1.0-beta.1"))]
        @test !haskey(entry(), "lazy")                                       # downloaded with the package
        ok, out, _ = bind(v1)
        @test ok && occursin("already binds fisp600-rank16 v0.1.0-beta.1", out)

        v2 = release(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.1, 7.0]"), "0.1.0-beta.2")
        ok, _, _ = bind(v2)
        @test ok && entry()["download"][1]["url"] == url("0.1.0-beta.2")
        before = read(toml)
        truncated = joinpath(dir, "truncated.tar.gz")
        write(truncated, read(v2)[1:1000])
        cases = [  # a release file to refuse => words that the refusal must contain
            release(s -> replace(s, "B1 = [0.5, 1.5]" => "B1 = [0.6, 1.5]"), "0.1.0-beta.3") =>
                "domain B1 [0.5, 1.5] → [0.6, 1.5]",
            release(identity, "0.1.0-alpha.1") => "version 0.1.0-alpha.1 does not follow version 0.1.0-beta.2",
            release(identity, "0.1.0-beta.4"; files = b -> write(joinpath(b, "stray.txt"), "not named by the card")) =>
                "files its card does not name: stray.txt",
            release(identity, "0.1.0-beta.5"; name = basename(url("0.1.0-beta.6"))) =>
                "must be named $(basename(url("0.1.0-beta.5")))",
            release(s -> replace(s, "schema_version = \"1.0\"" => "schema_version = \"9.0\""), "0.1.0-beta.7") =>
                "has schema_version 9.0",
            # The file is in the depot of this machine, but not in the release
            release(s -> replace(s, "\"rf_data/rf_phase_list.txt\"" => "\"../rf_phase_list.txt\""), "0.1.0-beta.9") =>
                "names ../rf_phase_list.txt, which is outside the bundle",
            truncated => "not a gzip-compressed tarball",
        ]
        for (file, words) in cases
            ok, _, err = bind(file)
            @test !ok && occursin(words, err)
            @test read(toml) == before                                      # unchanged
        end

        # Pkg downloads what is bound: the entry of v2, with the local file in
        # place of its URL, installs into an empty depot, without the package
        # server and without the network.
        offline = joinpath(dir, "Offline.toml")
        write(offline, replace(read(toml, String), url("0.1.0-beta.2") => "file://$v2"))
        fresh = mkpath(joinpath(dir, "fresh"))
        code = "using Pkg.Artifacts; print(ensure_artifact_installed(\"neuralbloch_fisp600_rank16\", $(repr(offline))))"
        env = merge(ENV, Dict("JULIA_DEPOT_PATH" => fresh * ":", "JULIA_PKG_SERVER" => ""))
        delete!(env, "JULIA_LOAD_PATH")
        @test startswith(readchomp(setenv(`$(Base.julia_cmd()) --startup-file=no -e $code`, env)), fresh)

        # One time as the command in the README (read throws if the exit code is not 0)
        script = joinpath(pkgdir(NeuralBloch), "dev", "bind_bundle.jl")
        v8 = release(s -> replace(s, "T1 = [0.1, 6.0]" => "T1 = [0.1, 8.0]"), "0.1.0-beta.8")
        env = merge(ENV, Dict("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], ':')))
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $v8 $toml`
        @test occursin("Bound fisp600-rank16 v0.1.0-beta.8", read(setenv(cmd, env), String))
        @test entry()["download"][1]["url"] == url("0.1.0-beta.8")
    end
end
