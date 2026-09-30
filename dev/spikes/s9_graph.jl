# Spike S9 (2026-09-30): with S4's profile (MIN 1), does a CUDA graph pay off
# at the smallest batches?
#
# Appendix A found graph replay no faster than enqueueV3 at 10k–40k voxels:
# the graph only shortens host submission, which overlaps GPU execution. At a
# small batch GPU time shrinks while the host submission does not, so the
# graph could start to matter. Capture uses CUDACore's own graph API (driver
# level; nothing in C++). Latency: sync after every call, median of 50.
#
#     julia --project=dev dev/spikes/s9_graph.jl

include("common.jl")
check_onnx()

e = Eng(engine_file("p_1_14400_40000"; min = 1, opt = 14400, max = 40000))
println(rpad("N", 8), rpad("enqueueV3 ms", 14), rpad("host µs", 10), rpad("graph ms", 11), rpad("host µs", 10), "graph output bitwise equal")
for N in (1, 16, 64, 256, 1000, 4000, 14400)
    X = CuArray(sample_inputs(N; seed = N))
    outs = run!(e, X); synchronize()
    ref = Array.(outs)
    enq = latency(() -> enqueue!(e))
    foreach(A -> fill!(A, 0), outs); synchronize()
    graph = CUDACore.capture() do
        enqueue!(e) || error("enqueueV3 failed during capture")
    end
    exec = CUDACore.instantiate(graph)
    CUDACore.launch(exec); synchronize()
    same = all(Array(outs[k]) == ref[k] for k in 1:3)
    rep = latency(() -> CUDACore.launch(exec))
    @printf("%-8d%-14.3f%-10.1f%-11.3f%-10.1f%s\n", N, enq.ms, enq.host_us, rep.ms, rep.host_us, same)
end
close(e)
