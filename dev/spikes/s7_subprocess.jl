# Spike S7 (2026-09-30): the builder as a subprocess of a CUDA-using Julia.
#
# ① Does a build succeed while this process holds a CUDA context and memory,
#   live or merely cached in CUDA.jl's pool — and when it fails, how clearly?
# ② When the builder fails (bad ONNX) or is killed, what does Julia see, and
#   what is left on disk?
# ③ Can the GPU be handed to the child via CUDA_VISIBLE_DEVICES — by index
#   and by UUID — and what happens with a bad value?
#
#     julia --project=dev dev/spikes/s7_subprocess.jl

include("common.jl")
check_onnx()
isfile(BUILDER) || compile_builder()

gpu_free() = parse(Int, strip(read(`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`, String)))
out = joinpath(WORK, "s7.engine")

function try_build(; env = ENV, onnx = ONNX, kill_after = nothing)
    rm(out; force = true)
    cmd = setenv(`$BUILDER $onnx $out 1 14400 40000 -1 -1 0`, env)
    o, err = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = o, stderr = err); wait = false)
    if kill_after !== nothing
        sleep(kill_after); kill(p, Base.SIGKILL)
    end
    wait(p)
    e = String(take!(err))
    lines = filter(!isempty, split(e, '\n'))
    return (; ok = success(p), exitcode = p.exitcode, signal = p.termsignal,
            file = isfile(out) ? filesize(out) : nothing, lastlines = last(lines, min(2, length(lines))))
end
show_result(label, r) = println("   ", rpad(label, 48), r.ok ? "ok" : "FAILED (exit $(r.exitcode), signal $(r.signal))",
                                 "   engine file: ", something(r.file, "none"),
                                 isempty(r.lastlines) ? "" : "\n      stderr: " * join(r.lastlines, "\n              "))

println("== ① parent holds CUDA state")
show_result("no CUDA in parent", try_build())
CUDACore.context()
show_result("parent has a context (free $(gpu_free()) MiB)", try_build())
live = CuArray{UInt8}(undef, 9 * 2^30)
show_result("parent holds 9 GiB live (free $(gpu_free()) MiB)", try_build())
live2 = CuArray{UInt8}(undef, round(Int, 1.1 * 2^30))
show_result("parent holds 10.1 GiB live (free $(gpu_free()) MiB)", try_build())
CUDACore.unsafe_free!(live); CUDACore.unsafe_free!(live2); live = live2 = nothing
synchronize()
show_result("…freed, still cached in the pool (free $(gpu_free()) MiB)", try_build())
CUDACore.reclaim()
show_result("after CUDACore.reclaim() (free $(gpu_free()) MiB)", try_build())

println("\n== ② builder failures")
bad = joinpath(WORK, "corrupt.onnx")
write(bad, read(ONNX)[1:100_000])
show_result("truncated ONNX", try_build(onnx = bad))
write(bad, "this is not an ONNX file")
show_result("garbage ONNX", try_build(onnx = bad))
show_result("killed (SIGKILL) after 2 s", try_build(kill_after = 2.0))

println("\n== ③ CUDA_VISIBLE_DEVICES")
uuid = "GPU-" * string(CUDACore.uuid(device()))
smi = strip(read(`nvidia-smi --query-gpu=uuid --format=csv,noheader`, String))
println("   CUDACore.uuid → $uuid   nvidia-smi → $smi   match: ", uuid == smi)
for v in ("0", uuid, "1", "GPU-00000000-0000-0000-0000-000000000000")
    show_result("CUDA_VISIBLE_DEVICES=$(length(v) > 12 ? v[1:12] * "…" : v)",
                try_build(env = merge(ENV, Dict("CUDA_VISIBLE_DEVICES" => v))))
end
