# The ground truth the scripts compare against: BlochSimulators 0.9's FISP3D
# EPG simulation with forward-difference derivatives, the generator of the
# training data, on the training sequence as the model card describes it.
# Included by scripts/validate.jl and dev/compare_mrstat.jl.

using NeuralBloch: Model, sequence
using BlochSimulators: BlochSimulators, FISP3D, @parameters, gpu, f32, simulate_magnetization,
                       simulate_derivatives_finite_difference
using ComputationalResources: CUDALibs
using CUDACore: CuArray
using Random: Random

# MRSTAT.jl's evaluation set (scripts/validate_tensorrt.jl): T₁ and T₂
# log-uniform over the training ranges, T₂ capped at 0.8 T₁, B₁ uniform.
# Float64, as the reference simulates it.
function evaluation_set(n = 14400; seed = 20260826)
    Random.seed!(seed)
    T₁ = exp.(rand(n) .* (log(5.0) - log(0.1)) .+ log(0.1))
    T₂ = min.(exp.(rand(n) .* (log(2.0) - log(0.01)) .+ log(0.01)), 0.8 .* T₁)
    B₁ = 0.8 .+ 0.4 .* rand(n)
    return (; T₁, T₂, B₁)
end

gpu_voxels(v) = map(x -> CuArray(Float32.(x)), v)

# Signal and derivatives for the voxels `v`, simulated as upstream generated
# the training data (data/data_generation.jl): on the GPU in Float32, with
# the card's forward-difference steps.
function epg(model::Model, v)
    s = sequence(model)
    rf = s.rf_angle .* exp.(im .* deg2rad.(s.rf_phase))        # degrees, phases in radians
    fisp = FISP3D(rf, s.TR, s.TE, s.max_state, s.TI, s.TW, s.repetitions, s.inversion_prepulse,
                  s.wait_spoiling, s.undersampling_factor, s.delta_k_spoil)
    T₁, T₂, B₁ = v.T₁, v.T₂, v.B₁
    parameters = gpu(f32(@parameters T₁ T₂ B₁))
    seq = gpu(f32(fisp))
    echos = simulate_magnetization(CUDALibs(), seq, parameters)
    d = model.derivatives
    d.method == "forward difference" || error("the reference implements forward differences, the card says $(d.method)")
    steps = BlochSimulators.T₁T₂B₁B₀(d.dT1_step, d.dT2_step, BlochSimulators.DEFAULT_ΔB₁, BlochSimulators.DEFAULT_ΔB₀)
    ∂ = simulate_derivatives_finite_difference((:T₁, :T₂), echos, seq, parameters, steps)
    return (signal = echos, ∂T₁ = ∂.T₁, ∂T₂ = ∂.T₂)
end

# Normalized mean absolute error, in Float64 on the CPU.
nmae(pred, ref) = (a = ComplexF64.(Array(pred)); b = ComplexF64.(Array(ref)); sum(abs, a .- b) / sum(abs, b))
