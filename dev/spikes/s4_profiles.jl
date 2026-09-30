# Spike S4 (2026-09-30): what does the profile's width cost?
#
# With a very wide profile (1–65536) instead of upstream's 10000–40000, do
# inference latency, the execution context's scratch memory, or build time
# change? The variants separate the effect of MIN, MAX and OPT. Outputs are
# compared against the upstream-profile engine, which also shows how far two
# different builds of the same ONNX can drift apart (test tolerances).
# Latency: device sync after every call, median of 50 (appendix A's method).
#
#     julia --project=dev dev/spikes/s4_profiles.jl

include("common.jl")
check_onnx()

profiles = [(10000, 14400, 40000), (1, 14400, 40000), (10000, 14400, 65536),
            (1, 14400, 65536), (1, 1000, 65536)]
println("== build (skipped when .work/spikes already has the engine)")
files = map(profiles) do (lo, opt, hi)
    name = (lo, opt, hi) == (10000, 14400, 40000) ? "upstream_settings" : "p_$(lo)_$(opt)_$(hi)"
    engine_file(name; min = lo, opt, max = hi)
end
# Build time on its own, for every profile: three fresh builds each, median.
buildtimes = map(profiles) do (lo, opt, hi)
    median(parse(Float64, build(joinpath(WORK, "scratch.engine"); min = lo, opt, max = hi)[1]["build_s"]) for _ in 1:3)
end

engines = Eng.(files)
ref = engines[1]
Ns = (1000, 14400, 40000, 65536)
inputs = Dict(N => CuArray(sample_inputs(N; seed = N)) for N in Ns)
refout = Dict(N => run!(ref, inputs[N]) for N in (14400, 40000))

println("\n", rpad("profile MIN/OPT/MAX", 22), rpad("build s", 9), rpad("scratch MiB", 13),
        join((rpad("N=$N ms", 11) for N in Ns)), "vs upstream-profile engine (max rel, worst voxel)")
for (p, e, bt) in zip(profiles, engines, buildtimes)
    lo, _, hi = p
    lat = map(Ns) do N
        lo ≤ N ≤ hi || return "—"
        outs = run!(e, inputs[N])                 # binds shape + addresses for this N
        @sprintf("%.3f", latency(() -> enqueue!(e)).ms)
    end
    cmp = map((14400, 40000)) do N
        o = run!(e, inputs[N]); synchronize()
        errs = [relerr(o[k], refout[N][k]).pervoxel for k in 1:3]
        all(iszero, errs) ? "N=$N bitwise" : @sprintf("N=%d %.1e/%.1e/%.1e", N, errs...)
    end
    println(rpad(join(p, "/"), 22), rpad(@sprintf("%.2f", bt), 9), rpad(device_memory(e) >> 20, 13),
            join(rpad.(lat, 11)), join(cmp, "   "))
end
foreach(close, engines)

# Second pass: the latency differences above are near the noise floor, so
# measure the candidate defaults again, interleaved round by round (7 rounds
# of 50 calls each) so clock and thermal drift hit all engines alike.
println("\n== interleaved re-measurement: median over 7 rounds of the per-round median (ms)")
cands = [(10000, 14400, 40000), (1, 14400, 40000), (1, 14400, 65536)]
ce = [Eng(files[findfirst(==(p), profiles)]) for p in cands]
for N in (14400, 40000)
    X = inputs[N]
    foreach(e -> run!(e, X), ce)
    rounds = [[(run!(e, X); latency(() -> enqueue!(e)).ms) for e in ce] for _ in 1:7]
    meds = [median(r[i] for r in rounds) for i in eachindex(ce)]
    spread = [extrema(r[i] for r in rounds) for i in eachindex(ce)]
    println("   N = $N: ", join((@sprintf("%s %.3f (%.3f–%.3f)", join(p, "/"), m, s...) for (p, m, s) in zip(cands, meds, spread)), "   "))
end
foreach(close, ce)
