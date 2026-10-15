# Accuracy against ground truth (T3): predictions against BlochSimulators'
# FISP3D EPG simulation, the generator of the training data, with the
# derivative method the card states, on the training sequence and domain the
# card describes. The bounds are twice what this fixed set measured
# (dev/NOTES.md, task 10): a convention error (layout, units, complex
# pairing) misses by ≈ 100 %. scripts/validate.jl reports the same on a
# larger set.

using NeuralBloch: Engine, Model, predict, domain, sequence
using CUDACore: CuArray
using Random: Xoshiro
using StructArrays: StructArray

# BlochSimulators is loaded only where T3 can run, so nothing below may use
# its macros: they would have to expand while this file is included anyway.
if blocker(:T3) === nothing
    using BlochSimulators: BlochSimulators, FISP3D, gpu, f32, simulate_magnetization,
                           simulate_derivatives_finite_difference, simulate_derivatives_forward_sensitivity
    using ComputationalResources: CUDALibs
end

# Measured 1.1837 / 0.6952 / 1.1449 % on this set for fisp600-rank16 v1 and
# again for 0.1.0-beta.1, the same network, against forward sensitivity
# (2026-10-15), doubled.
const ACCURACY_BOUNDS = (signal = 0.0237, ∂T₁ = 0.0140, ∂T₂ = 0.0229)

# A fixed set inside the card's training domain, sampled as upstream samples
# it: T₁, T₂ log-uniform, B₁ uniform, T₂ < T₁ (here capped at 0.8 T₁).
function accuracy_set(model, n = 2000; seed = 2026)
    rng = Xoshiro(seed)
    (T₁lo, T₁hi), (T₂lo, T₂hi), (B₁lo, B₁hi) = domain(model)
    T₁ = exp.(rand(rng, n) .* (log(T₁hi) - log(T₁lo)) .+ log(T₁lo))
    T₂ = min.(exp.(rand(rng, n) .* (log(T₂hi) - log(T₂lo)) .+ log(T₂lo)), 0.8 .* T₁)
    B₁ = B₁lo .+ (B₁hi - B₁lo) .* rand(rng, n)
    return (; T₁, T₂, B₁)
end

function epg_reference(model, v)
    s = sequence(model)
    rf = s.rf_angle .* exp.(im .* deg2rad.(s.rf_phase))            # degrees, phases in radians
    seq = gpu(f32(FISP3D(rf, s.TR, s.TE, s.max_state, s.TI, s.TW, s.repetitions, s.inversion_prepulse,
                         s.wait_spoiling, s.undersampling_factor, s.delta_k_spoil)))
    parameters = gpu(f32(StructArray{BlochSimulators.T₁T₂B₁{Float64}}((v.T₁, v.T₂, v.B₁))))  # @parameters T₁ T₂ B₁
    d = model.derivatives
    if d.method == "forward sensitivity"
        echos, ∂ = simulate_derivatives_forward_sensitivity((:T₁, :T₂), seq, parameters)
    else                                                                # forward difference, Model's only other method
        echos = simulate_magnetization(CUDALibs(), seq, parameters)
        steps = BlochSimulators.T₁T₂B₁B₀(d.dT1_step, d.dT2_step, BlochSimulators.DEFAULT_ΔB₁,
                                         BlochSimulators.DEFAULT_ΔB₀)
        ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, seq, parameters, steps)
    end
    return (signal = echos, ∂T₁ = ∂.T₁, ∂T₂ = ∂.T₂)
end

nmae(pred, ref) = (a = ComplexF64.(Array(pred)); b = ComplexF64.(Array(ref)); sum(abs, a .- b) / sum(abs, b))

tier(:T3, "accuracy: NMAE against EPG") do
    model = Model()
    engine = Engine(model)
    v = accuracy_set(model)
    pred = predict(engine, map(x -> CuArray(Float32.(x)), v))
    ref = epg_reference(model, v)
    for k in (:signal, :∂T₁, :∂T₂)
        e = nmae(pred[k], ref[k])
        @info "NMAE against EPG" output = k value = round(100e; digits = 4) bound = 100ACCURACY_BOUNDS[k]
        @test e ≤ ACCURACY_BOUNDS[k]
    end
    close(engine)
end
