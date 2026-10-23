# The ground truth to compare with: the FISP3D EPG simulation of
# BlochSimulators. This simulation made the training data. The derivative
# method, the sequence and the training domain come from the model card.
# scripts/validate.jl, scripts/golden.jl and dev/compare_mrstat.jl include
# this file.

using NeuralBloch: Model, domain, sequence
using BlochSimulators: BlochSimulators, FISP3D, @parameters, gpu, f32, simulate_magnetization,
                       simulate_derivatives_finite_difference, simulate_derivatives_forward_sensitivity
using ComputationalResources: CUDALibs
using CUDACore: CuArray
using Random: Xoshiro

# A set of voxels in the training domain of the card. T₁ and T₂ are
# log-uniform and B₁ is uniform, as upstream. Upstream drops the voxels with
# T₂ ≥ T₁. This set instead limits T₂ to 0.8 T₁, as the evaluation set of
# MRSTAT.jl does: T₂ < T₁ in the domain. The values are Float64, as the reference
# uses. Xoshiro(seed) gives the same values as Random.seed!(seed). Thus, on the
# domain of the 2026-10-01 card (T₁ 0.1–5 s, B₁ 0.8–1.2), the default set is
# the evaluation set of MRSTAT.jl (scripts/validate_tensorrt.jl).
function evaluation_set(model::Model, n = 14400; seed = 20260826)
    rng = Xoshiro(seed)
    (T₁lo, T₁hi), (T₂lo, T₂hi), (B₁lo, B₁hi) = domain(model)
    T₁ = exp.(rand(rng, n) .* (log(T₁hi) - log(T₁lo)) .+ log(T₁lo))
    T₂ = min.(exp.(rand(rng, n) .* (log(T₂hi) - log(T₂lo)) .+ log(T₂lo)), 0.8 .* T₁)
    B₁ = B₁lo .+ (B₁hi - B₁lo) .* rand(rng, n)
    return (; T₁, T₂, B₁)
end

gpu_voxels(v) = map(x -> CuArray(Float32.(x)), v)

# The signal and its derivatives for the voxels `v`. The simulation is the same
# as the one that made the training data upstream (data/data_generation.jl):
# on the GPU, in Float32, with the derivative method of the card.
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
    elseif d.method == "forward difference"
        echos = simulate_magnetization(CUDALibs(), seq, parameters)
        steps = BlochSimulators.T₁T₂B₁B₀(d.dT1_step, d.dT2_step, BlochSimulators.DEFAULT_ΔB₁,
                                         BlochSimulators.DEFAULT_ΔB₀)
        ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, seq, parameters, steps)
    else
        error("epg does not know the derivative method $(repr(d.method)) of the card")
    end
    return (signal = echos, ∂T₁ = ∂.T₁, ∂T₂ = ∂.T₂)
end

# The normalized mean absolute error, in Float64 on the CPU.
nmae(pred, ref) = (a = ComplexF64.(Array(pred)); b = ComplexF64.(Array(ref)); sum(abs, a .- b) / sum(abs, b))
