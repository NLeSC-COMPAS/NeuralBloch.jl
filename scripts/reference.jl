# The ground truth the scripts compare against: BlochSimulators' FISP3D EPG
# simulation, the generator of the training data, with the derivative method
# the model card states (forward sensitivity, or forward differences with the
# card's steps), on the training sequence and domain as the card describes
# them. Included by scripts/validate.jl and dev/compare_mrstat.jl.

using NeuralBloch: Model, domain, sequence
using BlochSimulators: BlochSimulators, FISP3D, @parameters, gpu, f32, simulate_magnetization,
                       simulate_derivatives_finite_difference, simulate_derivatives_forward_sensitivity
using ComputationalResources: CUDALibs
using CUDACore: CuArray
using Random: Random

# An evaluation set over the card's training domain, sampled as upstream
# samples it: T₁ and T₂ log-uniform, B₁ uniform, with T₂ capped at 0.8 T₁
# (the domain has T₂ < T₁). Float64, as the reference simulates it. On the
# domain of the 2026-10-01 card (T₁ 0.1–5 s, B₁ 0.8–1.2) this is MRSTAT.jl's
# evaluation set (scripts/validate_tensorrt.jl).
function evaluation_set(model::Model, n = 14400; seed = 20260826)
    Random.seed!(seed)
    (T₁lo, T₁hi), (T₂lo, T₂hi), (B₁lo, B₁hi) = domain(model)
    T₁ = exp.(rand(n) .* (log(T₁hi) - log(T₁lo)) .+ log(T₁lo))
    T₂ = min.(exp.(rand(n) .* (log(T₂hi) - log(T₂lo)) .+ log(T₂lo)), 0.8 .* T₁)
    B₁ = B₁lo .+ (B₁hi - B₁lo) .* rand(n)
    return (; T₁, T₂, B₁)
end

gpu_voxels(v) = map(x -> CuArray(Float32.(x)), v)

# Signal and derivatives for the voxels `v`, simulated as upstream generated
# the training data (data/data_generation.jl): on the GPU in Float32, with the
# card's derivative method.
function epg(model::Model, v)
    s = sequence(model)
    rf = s.rf_angle .* exp.(im .* deg2rad.(s.rf_phase))        # degrees, phases in radians
    fisp = FISP3D(rf, s.TR, s.TE, s.max_state, s.TI, s.TW, s.repetitions, s.inversion_prepulse,
                  s.wait_spoiling, s.undersampling_factor, s.delta_k_spoil)
    T₁, T₂, B₁ = v.T₁, v.T₂, v.B₁
    parameters = gpu(f32(@parameters T₁ T₂ B₁))
    seq = gpu(f32(fisp))
    d = model.derivatives
    if d.method == "forward sensitivity"
        echos, ∂ = simulate_derivatives_forward_sensitivity((:T₁, :T₂), seq, parameters)
    else                                                        # forward difference, Model's only other method
        echos = simulate_magnetization(CUDALibs(), seq, parameters)
        steps = BlochSimulators.T₁T₂B₁B₀(d.dT1_step, d.dT2_step, BlochSimulators.DEFAULT_ΔB₁,
                                         BlochSimulators.DEFAULT_ΔB₀)
        ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, seq, parameters, steps)
    end
    return (signal = echos, ∂T₁ = ∂.T₁, ∂T₂ = ∂.T₂)
end

# Normalized mean absolute error, in Float64 on the CPU.
nmae(pred, ref) = (a = ComplexF64.(Array(pred)); b = ComplexF64.(Array(ref)); sum(abs, a .- b) / sum(abs, b))
