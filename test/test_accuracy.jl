# Accuracy against the ground truth (T3): the predictions against the FISP3D
# EPG simulation of BlochSimulators, the simulation that made the training
# data, in scripts/reference.jl. The bounds are two times the values that this
# fixed set measured (dev/NOTES.md, task 10). An error in a convention (layout,
# units, complex pairs) gives an error of approximately 100 %.
# scripts/validate.jl measures the same on a larger set.

using NeuralBloch: Engine, Model, predict

# Load BlochSimulators only where T3 can run. include runs when the test runs,
# thus the macros of BlochSimulators in reference.jl expand after this load.
if blocker(:T3) === nothing
    include(joinpath(@__DIR__, "..", "scripts", "reference.jl"))
end

# Measured 1.1837 / 0.6952 / 1.1449 % on this set for fisp600-rank16 v1 and
# again for 0.1.0-beta.1, the same network, against forward sensitivity
# (2026-10-15), multiplied by two.
const ACCURACY_BOUNDS = (signal = 0.0237, ∂T₁ = 0.0140, ∂T₂ = 0.0229)

tier(:T3, "accuracy: NMAE against EPG") do
    model = Model()
    engine = Engine(model)
    v = evaluation_set(model, 2000; seed = 2026)
    pred = predict(engine, gpu_voxels(v))
    ref = epg(model, v)
    for k in (:signal, :∂T₁, :∂T₂)
        e = nmae(pred[k], ref[k])
        @info "NMAE against EPG" output = k value = round(100e; digits = 4) bound = 100ACCURACY_BOUNDS[k]
        @test e ≤ ACCURACY_BOUNDS[k]
    end
    close(engine)
end
