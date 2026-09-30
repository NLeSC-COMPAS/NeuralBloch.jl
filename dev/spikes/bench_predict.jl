# Spike (2026-09-30): predict!-style vs predict-style calls.
#
# Question: how much faster is writing into caller-owned fixed buffers than
# returning fresh outputs per call, and where does the difference come from?
# Results: PLAN.md, appendix A (first table and the GC numbers).
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
#     julia --project=../MRSTAT.jl dev/spikes/bench_predict.jl
#
#   A  predict!            : fixed buffers, graph replay, no alloc, no copy
#   B  predict (copy)      : graph replay into fixed buffers + alloc + copy ×3   (today's compute_echos_and_derivatives)
#   C  predict (rebind)    : alloc ×3 + setTensorAddress ×3 + enqueueV3 (no graph, no copy)
#   D  enqueueV3 fixed     : fixed buffers, enqueueV3, no alloc  (isolates the graph effect)
#
# Latency = wall time per call with a sync after each call (what the solver sees:
# every objective evaluation ends in a host read). Also a 200-call loop without
# per-call sync for the throughput / GC picture.

using MRSTAT, BlochSimulators, CUDA, StructArrays, Statistics, Printf, Random
using MRSTAT.ForwardModels: _stage!, _infer!, _bind!, _devptr, _streamptr, _lasterr

m = load_tensorrt_model()
api, h = m.api, m.h

function params(n)
    Random.seed!(1)
    T₁ = exp.(rand(n) .* (log(5.0) - log(0.1)) .+ log(0.1))
    T₂ = min.(exp.(rand(n) .* (log(2.0) - log(0.01)) .+ log(0.01)), 0.8 .* T₁)
    B₁ = 0.8 .+ 0.4 .* rand(n)
    gpu(f32(StructArray(map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ones(n), zeros(n)))))
end

setaddr(name, A) = (rc = ccall(api.set_address, Int32, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), h, name, _devptr(A));
                    rc == 0 || error(_lasterr(m)))
enqueue() = (rc = ccall(api.enqueue, Int32, (Ptr{Cvoid}, Ptr{Cvoid}), h, _streamptr());
             rc == 0 || error(_lasterr(m)))

A!(p) = (_stage!(m, p); _infer!(m); nothing)
B(p)  = (_stage!(m, p); _infer!(m); (copy(m.raw_signal), copy(m.raw_dT1), copy(m.raw_dT2)))
function C(p)
    _stage!(m, p)
    o = ntuple(_ -> CuMatrix{ComplexF32}(undef, m.NTR, length(p)), 3)
    setaddr("signal", o[1]); setaddr("dT1", o[2]); setaddr("dT2", o[3])
    enqueue()
    return o
end
D!(p) = (_stage!(m, p); enqueue(); nothing)

function lat(f, p; n = 30)
    f(p); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:n
        CUDA.synchronize()
        t = @elapsed (f(p); CUDA.synchronize())
        push!(ts, 1e3t)
    end
    return median(ts), maximum(ts)
end

function loop(f, p; n = 200)
    f(p); CUDA.synchronize(); GC.gc(true); CUDA.reclaim()
    gc0 = Base.gc_num()
    t = @elapsed begin
        for _ in 1:n; f(p); end
        CUDA.synchronize()
    end
    gcd = Base.GC_Diff(Base.gc_num(), gc0)
    return 1e3t / n, gcd.total_time / 1e6, gcd.pause
end

rebind!() = (setaddr("signal", m.raw_signal); setaddr("dT1", m.raw_dT1); setaddr("dT2", m.raw_dT2))

println("GPU: ", CUDA.name(CUDA.device()), "   NTR = ", m.NTR,
        "   profile = [", m.minB, ", ", m.maxB, "]")
for N in (m.minB, m.optB, m.maxB)
    p = params(N)
    m.N = 0; _bind!(m, N)                         # fresh binding + graph for this N
    @assert m.graph_ok
    mb = 3 * m.NTR * N * 8 / 2^20
    @printf("\nN = %d   (outputs: 3 × %d × %d ComplexF32 = %.0f MiB per call)\n", N, m.NTR, N, mb)
    @printf("  %-34s %10s %10s   %12s %10s %6s\n", "", "median ms", "max ms", "loop ms/call", "GC ms", "GCs")
    for (name, f) in (("A predict!  (graph, no alloc/copy)", A!),
                      ("B predict   (graph + alloc + copy)", B),
                      ("D enqueueV3, fixed buffers", D!),
                      ("C predict   (alloc + rebind + enq)", C))
        med, mx = lat(f, p)
        per, gct, np = loop(f, p)
        f === C && rebind!()                      # restore fixed-buffer addresses
        @printf("  %-34s %10.2f %10.2f   %12.2f %10.1f %6d\n", name, med, mx, per, gct, np)
    end
end
