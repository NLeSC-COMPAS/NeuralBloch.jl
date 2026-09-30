# Spike S5 (2026-09-30): what does one build cost, and should the workspace
# be capped?
#
# TensorRT's default workspace limit is the whole device memory. Build with
# the default and with caps; record the builder's wall time, its peak GPU
# memory (nvidia-smi sampled every 20 ms, minus the idle baseline), its peak
# host RSS (GNU time), and what the cap does to the engine: scratch memory,
# latency, outputs. Runs before this process touches the GPU, so the samples
# are the builder's alone.
#
#     julia --project=dev dev/spikes/s5_build_cost.jl

include("common.jl")
check_onnx()
isfile(BUILDER) || compile_builder()

gpu_used() = parse(Int, strip(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)))

function measured_build(out, workspace_mib)
    base = gpu_used()
    samples = Int[]
    sampler = open(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 20`)
    reader = @async for l in eachline(sampler); push!(samples, parse(Int, strip(l))); end
    sleep(0.3)
    cmd = `/usr/bin/time -f "maxrss_kb = %M" $BUILDER $ONNX $out 1 14400 40000 -1 -1 $workspace_mib`
    out_io, err_io = IOBuffer(), IOBuffer()
    t = @elapsed (p = run(pipeline(ignorestatus(cmd); stdout = out_io, stderr = err_io)))
    sleep(0.3)
    kill(sampler); wait(reader)
    text, err = String(take!(out_io)), String(take!(err_io))
    peak = maximum(samples) - base
    if !success(p)
        m = match(r"Error Code[^\n]*", err)
        return (; wall = t, build = NaN, peak, rss = 0, error = m === nothing ? err : m.match)
    end
    kv = Dict(strip(k) => strip(v) for (k, v) in (split(l, " = ") for l in split(chomp(text), '\n')))
    rss = parse(Int, match(r"maxrss_kb = (\d+)", err)[1]) ÷ 1024
    return (; wall = t, build = parse(Float64, kv["build_s"]), peak, rss, error = "")
end

caps = [0, 1024, 512, 448, 440, 400, 256]          # MiB; 0 = TensorRT's default
println("== builds with profile 1/14400/40000 (3 runs each; median wall/build time, max peak)")
println(rpad("workspace", 12), rpad("wall s", 9), rpad("build s", 9), rpad("peak GPU MiB", 14), "peak RSS MiB")
built = String[]
for cap in caps
    out = joinpath(WORK, "ws_$cap.engine")
    runs = [measured_build(out, cap) for _ in 1:3]
    label = rpad(cap == 0 ? "default" : "$cap MiB", 12)
    if !isempty(runs[1].error)
        println(label, "build fails: ", runs[1].error)
        continue
    end
    println(label, rpad(@sprintf("%.2f", median(r.wall for r in runs)), 9),
            rpad(@sprintf("%.2f", median(r.build for r in runs)), 9),
            rpad(maximum(r.peak for r in runs), 14), maximum(r.rss for r in runs))
    push!(built, out)
end

println("\n== the engines those caps produced")
X = CuArray(sample_inputs(14400))
engines = Eng.(built)
ref = run!(engines[1], X)
for (path, e) in zip(built, engines)
    o = run!(e, X); synchronize()
    same = all(Array(o[k]) == Array(ref[k]) for k in 1:3)
    errs = [relerr(o[k], ref[k]).pervoxel for k in 1:3]
    @printf("   %-12s scratch %4d MiB   N=14400 %.3f ms   vs default: %s\n",
            basename(path), device_memory(e) >> 20, latency(() -> enqueue!(e)).ms,
            same ? "bitwise" : @sprintf("max rel %.1e/%.1e/%.1e", errs...))
end
foreach(close, engines)
