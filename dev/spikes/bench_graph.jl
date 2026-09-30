# Spike (2026-09-30): does the CUDA graph pay off, and what does an output copy cost?
#
# Question: at the engine's batch sizes, is graph replay faster than a plain
# enqueueV3 (with fixed buffers, and when alternating between two buffer sets
# the way MRSTAT's two workspaces would)? How long does the host take to
# submit each, and how long does copying the three outputs take?
# Results: PLAN.md, appendix A (second table).
#
# This measures MRSTAT.jl's existing TensorRT engine through MRSTAT's internal
# (non-public) functions, so it is a record of that measurement, not maintained
# code. Once NeuralBloch.jl has its own engine, redo it against its API
# (spikes S4 and S9).
#
# Run from this repository's root, against a MRSTAT.jl checkout whose engine
# (models/model.engine) and shim (deps/tensorrt/libtrtjl.so) are in place.
# Source MRSTAT's env.sh first. Loading MRSTAT pulls in PythonPlot, and without
# env.sh CondaPkg builds its own ~1.3 GB Python environment in
# ../MRSTAT.jl/.CondaPkg.
#
#     source ../MRSTAT.jl/env.sh
#     julia --project=../MRSTAT.jl dev/spikes/bench_graph.jl

using MRSTAT, BlochSimulators, CUDA, StructArrays, Statistics, Printf, Random
using MRSTAT.ForwardModels: _stage!, _infer!, _bind!, _devptr, _streamptr, _lasterr
using Logging; m = with_logger(NullLogger()) do; load_tensorrt_model(); end
api, h = m.api, m.h
function params(n)
    Random.seed!(1)
    T₁ = exp.(rand(n) .* (log(5.0) - log(0.1)) .+ log(0.1))
    T₂ = min.(exp.(rand(n) .* (log(2.0) - log(0.01)) .+ log(0.01)), 0.8 .* T₁)
    B₁ = 0.8 .+ 0.4 .* rand(n)
    gpu(f32(StructArray(map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ones(n), zeros(n)))))
end
setaddr(name, A) = (rc = ccall(api.set_address, Int32, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), h, name, _devptr(A)); rc == 0 || error(_lasterr(m)))
enqueue() = (rc = ccall(api.enqueue, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), h, _streamptr()); rc == 0 || error(_lasterr(m)))
replay()  = (rc = ccall(api.replay, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), h, _streamptr()); rc == 0 || error(_lasterr(m)))
bind3(o) = (setaddr("signal", o[1]); setaddr("dT1", o[2]); setaddr("dT2", o[3]))
function lat(f; n = 50)
    f(); CUDA.synchronize(); ts = Float64[]
    for _ in 1:n
        CUDA.synchronize(); t = @elapsed (f(); CUDA.synchronize()); push!(ts, 1e3t)
    end
    median(ts)
end
function hostsubmit(f; n = 50)   # host time of the submission alone, queue empty
    f(); CUDA.synchronize(); ts = Float64[]
    for _ in 1:n
        CUDA.synchronize(); t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e3)
    end
    CUDA.synchronize(); median(ts)
end
for N in (m.minB, m.optB, m.maxB)
    p = params(N); m.N = 0; _bind!(m, N); @assert m.graph_ok
    W1 = (m.raw_signal, m.raw_dT1, m.raw_dT2)
    W2 = ntuple(_ -> CuMatrix{ComplexF32}(undef, m.NTR, N), 3)
    k = Ref(0)
    alt() = (k[] += 1; bind3(isodd(k[]) ? W1 : W2); enqueue())
    _stage!(m, p); CUDA.synchronize()
    @printf("N = %d\n", N)
    @printf("  engine only, graph replay                 : %6.2f ms   host submit %7.1f µs\n", lat(replay), hostsubmit(replay))
    bind3(W1)
    @printf("  engine only, enqueueV3 fixed buffers      : %6.2f ms   host submit %7.1f µs\n", lat(enqueue), hostsubmit(enqueue))
    @printf("  engine only, enqueueV3 alternating 2 sets : %6.2f ms   host submit %7.1f µs\n", lat(alt), hostsubmit(alt))
    bind3(W1)
    @printf("  copy ×3 alone                             : %6.2f ms\n", lat(() -> (copyto!(W2[1], W1[1]); copyto!(W2[2], W1[2]); copyto!(W2[3], W1[3]))))
end
