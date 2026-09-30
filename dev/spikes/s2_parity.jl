# Spike S2 (2026-09-30): does our builder, with upstream's settings, reproduce
# MRSTAT's engine?
#
# Upstream's build_engine.py sets only the profile (10000/14400/40000) and
# leaves everything else at TensorRT 11.2.1.2 defaults. Build the same way,
# twice, run all engines on identical inputs drawn from the training
# distribution (T₂ < T₁), and compare. Also: is one engine run-to-run
# deterministic? Decides the test tolerances.
#
#     julia --project=dev dev/spikes/s2_parity.jl

include("common.jl")
check_onnx()

println("== build twice with upstream's settings (profile 10000/14400/40000, all else default)")
b1 = engine_file("upstream_settings"; min = 10000, opt = 14400, max = 40000)
b2 = engine_file("upstream_settings_2"; min = 10000, opt = 14400, max = 40000)
for (n, p) in (("build 1", b1), ("build 2", b2), ("MRSTAT", REF_ENGINE))
    println("   ", rpad(n, 8), ": ", filesize(p), " bytes, sha256 ", filesha(p)[1:12], "…")
end
kernels(p) = [m.match for m in eachmatch(r"kernel=\w+", String(read(p)))]
println("   GEMM kernels differing from MRSTAT's engine: build 1 ",
        count(splat(!=), zip(kernels(b1), kernels(REF_ENGINE))), ", build 2 ",
        count(splat(!=), zip(kernels(b2), kernels(REF_ENGINE))), " (of ", length(kernels(REF_ENGINE)), ")")

engines = [Eng(b1), Eng(b2), Eng(REF_ENGINE)]
println("   scratch: ", join((device_memory(e) >> 20 for e in engines), " / "), " MiB")

for (N, seed) in ((14400, 1), (10000, 2), (40000, 3))
    X = CuArray(sample_inputs(N; seed))
    o1, o2, ref = (run!(e, X) for e in engines)
    again = run!(engines[1], X)
    synchronize()
    println("\n   N = $N")
    for (k, name) in enumerate(("signal", "dT1", "dT2"))
        r1, r2 = relerr(o1[k], ref[k]), relerr(o2[k], ref[k])
        @printf("     %-6s vs MRSTAT: build 1 bitwise %-5s (max rel %.1e)   build 2 bitwise %-5s (max rel %.1e)   rerun bitwise: %s\n",
                name, Array(o1[k]) == Array(ref[k]), r1.pervoxel, Array(o2[k]) == Array(ref[k]), r2.pervoxel,
                Array(o1[k]) == Array(again[k]))
    end
end
foreach(close, engines)
