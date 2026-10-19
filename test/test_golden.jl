# The predictions against stored EPG values (T2). The other tests compare the
# engines with each other. Thus, an error that all the engines share (the
# order of T₁ and T₂ in x, Re and Im, the units of the derivatives) passes
# them. Such an error gives an NMAE of approximately 100 % here. The test
# accepts two times the NMAE that scripts/golden.jl measured when it wrote
# golden.toml. The values come from the EPG simulation of BlochSimulators
# (scripts/reference.jl), and they are for one model. scripts/validate.jl
# measures the accuracy on a large set.

using NeuralBloch: Engine, Model, predict, cardname
using CUDACore: CuArray
using TOML: TOML

tier(:T2, "golden: predictions against stored EPG values") do
    golden = TOML.parsefile(joinpath(@__DIR__, "golden.toml"))
    model = Model()
    stored, bound = "$(golden["family"]) v$(golden["version"])", "$(model.family) v$(model.version)"
    if stored != bound
        @error "test/golden.toml is for $stored, but Artifacts.toml binds $bound. " *
               "Run scripts/golden.jl, and examine the NMAE that it gives."
        @test stored == bound
        return
    end
    v = golden["voxels"]
    p = (T₁ = CuArray(Float32.(v["T1"])), T₂ = CuArray(Float32.(v["T2"])), B₁ = CuArray(Float32.(v["B1"])))
    engine = Engine(model)
    pred = predict(engine, p)
    rows = golden["readouts"]
    for k in (:signal, :∂T₁, :∂T₂)
        g = golden[cardname(k)]
        ref = complex.(stack(g["re"]), stack(g["im"]))           # (readouts, voxels)
        e = sum(abs, ComplexF64.(Array(pred[k])[rows, :]) .- ref) / sum(abs, ref)
        @test e ≤ 2g["nmae"]
    end
    close(engine)
end
