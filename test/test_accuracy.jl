# Accuracy against ground truth (T3): predictions against BlochSimulators
# 0.9's FISP3D EPG simulation with forward-difference derivatives, the
# generator of the training data, on the training sequence the card
# describes. The bounds are twice what this fixed set measured (dev/NOTES.md,
# task 10): a convention error (layout, units, complex pairing) misses by
# ≈ 100 %. scripts/validate.jl reports the same on MRSTAT.jl's larger set.

using NeuralBloch: Engine, Model, predict, sequence
using CUDACore: CuArray
using Random: Xoshiro

if blocker(:T3) === nothing
    using BlochSimulators: BlochSimulators, FISP3D, @parameters, gpu, f32, simulate_magnetization,
                           simulate_derivatives_finite_difference
    using ComputationalResources: CUDALibs
end

# Measured 0.5507 / 0.2485 / 0.4579 % on this set (2026-10-01), doubled.
const ACCURACY_BOUNDS = (signal = 0.0110, ∂T₁ = 0.0050, ∂T₂ = 0.0092)

# A fixed set inside the training domain, sampled as upstream samples it:
# T₁, T₂ log-uniform, B₁ uniform, T₂ < T₁ (here capped at 0.8 T₁).
function accuracy_set(n = 2000; seed = 2026)
    rng = Xoshiro(seed)
    T₁ = exp.(rand(rng, n) .* (log(5.0) - log(0.1)) .+ log(0.1))
    T₂ = min.(exp.(rand(rng, n) .* (log(2.0) - log(0.01)) .+ log(0.01)), 0.8 .* T₁)
    B₁ = 0.8 .+ 0.4 .* rand(rng, n)
    return (; T₁, T₂, B₁)
end

function epg_reference(model, v)
    s = sequence(model)
    rf = s.rf_angle .* exp.(im .* deg2rad.(s.rf_phase))            # degrees, phases in radians
    seq = gpu(f32(FISP3D(rf, s.TR, s.TE, s.max_state, s.TI, s.TW, s.repetitions, s.inversion_prepulse,
                         s.wait_spoiling, s.undersampling_factor, s.delta_k_spoil)))
    T₁, T₂, B₁ = v.T₁, v.T₂, v.B₁
    parameters = gpu(f32(@parameters T₁ T₂ B₁))
    echos = simulate_magnetization(CUDALibs(), seq, parameters)
    d = model.derivatives
    steps = BlochSimulators.T₁T₂B₁B₀(d.dT1_step, d.dT2_step, BlochSimulators.DEFAULT_ΔB₁, BlochSimulators.DEFAULT_ΔB₀)
    ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, seq, parameters, steps)
    return (signal = echos, ∂T₁ = ∂.T₁, ∂T₂ = ∂.T₂)
end

nmae(pred, ref) = (a = ComplexF64.(Array(pred)); b = ComplexF64.(Array(ref)); sum(abs, a .- b) / sum(abs, b))

tier(:T3, "accuracy: NMAE against EPG") do
    model = Model()
    engine = Engine(model)
    v = accuracy_set()
    pred = predict(engine, map(x -> CuArray(Float32.(x)), v))
    ref = epg_reference(model, v)
    for k in (:signal, :∂T₁, :∂T₂)
        e = nmae(pred[k], ref[k])
        @info "NMAE against EPG" output = k value = round(100e; digits = 4) bound = 100ACCURACY_BOUNDS[k]
        @test e ≤ ACCURACY_BOUNDS[k]
    end
    close(engine)
end
