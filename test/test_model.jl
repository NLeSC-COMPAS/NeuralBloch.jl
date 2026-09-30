# model.jl: the bundle (ONNX, card, RF tables) and its checks. Julia only.

using NeuralBloch: Model, BundleError

const BUNDLE = joinpath(pkgdir(NeuralBloch), "models")

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
    @test m.onnx == joinpath(BUNDLE, "model.onnx")
    @test m.sha256 == "3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607"
    @test m.ntr == 600
    @test m.domain == (T1 = (0.1, 5.0), T2 = (0.01, 2.0), B1 = (0.8, 1.2))
    @test m.constraint == "T2 < T1"
    @test m.derivatives == (method = "forward difference", dT1_step = 0.02, dT2_step = 0.0025)
    s = m.sequence
    @test (s.kind, s.TR, s.TE, s.TI, s.TW) == ("FISP3D", 0.00558, 0.002745, 0.01084, 0.0)
    @test (s.max_state, s.repetitions, s.undersampling_factor, s.delta_k_spoil) == (64, 5, 1, 0.0)
    @test s.inversion_prepulse && s.wait_spoiling
    @test length(s.rf_angle) == length(s.rf_phase) == 600
    @test (s.rf_angle[1], s.rf_angle[end]) == (3.960833, parse(Float64, last(readlines(joinpath(BUNDLE, "rf_data", "rf_angle_list.txt")))))
    @test s.rf_phase[1] == -1.9641
    @test m.provenance["upstream"] == "NeuralBloch@9f4db84"
    @test occursin("ntr = 600", sprint(show, m))
end

tier(:T0, "model: malformed cards") do
    cases = [  # edit of the card => words the error must contain
        (s -> replace(s, r"\[domain\].*?\n\n"s => "")) => "table [domain] is missing",
        (s -> replace(s, r"\nntr = 600[^\n]*" => "")) => "[model] ntr is missing",
        (s -> replace(s, "ntr = 600" => "ntr = \"600\"")) => "[model] ntr must be an integer",
        (s -> replace(s, "ntr = 600" => "ntr = 600.0")) => "[model] ntr must be an integer",
        (s -> replace(s, "T1 = [0.1, 5.0]" => "T1 = [5.0, 0.1]")) => "[domain] T1 must be two numbers [low, high] with low < high",
        (s -> replace(s, "T1 = [0.1, 5.0]" => "T1 = [0.1]")) => "[domain] T1 must be two numbers",
        (s -> replace(s, "TR = 0.00558" => "TR = \"5.58 ms\"")) => "[sequence] TR must be a number",
        (s -> replace(s, "inversion_prepulse = true" => "inversion_prepulse = \"yes\"")) => "[sequence] inversion_prepulse must be true or false",
        (s -> replace(s, "constraint = \"T2 < T1\"" => "constraint = \"T1 < T2\"")) => "unsupported constraint \"T1 < T2\"",
        (s -> replace(s, "sha256 = \"" => "sha256 = \"xyz")) => "[model] sha256 must be 64 hexadecimal digits",
        (s -> replace(s, "[model]" => "[model")) => "is not valid TOML",
        (s -> replace(s, "rf_data/rf_angle_list.txt" => "rf_data/missing.txt")) => "missing.txt, which",
        (s -> replace(s, "onnx = \"model.onnx\"" => "onnx = \"other.onnx\"")) => "other.onnx, which",
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

tier(:T0, "model: the ONNX must be the one the card pins") do
    mktempdir() do dir
        b = edited_bundle(identity, dir)
        write(joinpath(b, "model.onnx"), read(joinpath(b, "model.onnx"))[1:end-1])
        err = @test_throws BundleError Model(b)
        msg = sprint(showerror, err.value)
        @test occursin("sha256", msg)
        @test occursin("3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607", msg)
        @test occursin("model.onnx", msg)
    end
    @test_throws BundleError Model("/nonexistent/bundle")
end
