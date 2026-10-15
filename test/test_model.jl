# model.jl: the bundle (ONNX per target, card, RF tables) and its checks. Julia only.

using NeuralBloch: Model, BundleError, TARGETS

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
    for target in TARGETS
        mktempdir() do dir
            b = edited_bundle(identity, dir)
            file = basename(Model().onnx[target])
            write(joinpath(b, file), read(joinpath(b, file))[1:end-1])
            err = @test_throws BundleError Model(b)
            msg = sprint(showerror, err.value)
            @test occursin("pins $(PINNED[target]) for $target", msg)
            @test occursin(file, msg)
        end
    end
    @test_throws BundleError Model("/nonexistent/bundle")
end
