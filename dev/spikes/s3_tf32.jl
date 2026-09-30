# Spike S3 (2026-09-30): does TF32 matter?
#
# Upstream never turns TF32 off (TensorRT enables it by default). The 2080 Ti
# (SM 7.5) has no TF32 hardware, so on this machine the only question we can
# answer is whether on and off give the same engine behaviour. The accuracy
# and speed comparison needs an Ampere-or-newer GPU and stays open.
#
#     julia --project=dev dev/spikes/s3_tf32.jl

include("common.jl")
check_onnx()

println("GPU: ", CUDACore.name(device()), " (SM ", CUDACore.capability(device()), ")")
on = engine_file("upstream_settings"; min = 10000, opt = 14400, max = 40000)          # tf32 default = on
off = engine_file("tf32_off"; min = 10000, opt = 14400, max = 40000, tf32 = 0)
kernels(p) = [m.match for m in eachmatch(r"kernel=\w+", String(read(p)))]
println("   engine bytes: on ", filesize(on), ", off ", filesize(off))
println("   kernels mentioning tf32: on ", count(contains("tf32"), kernels(on)),
        ", off ", count(contains("tf32"), kernels(off)))

eon, eoff = Eng(on), Eng(off)
for N in (10000, 14400, 40000)
    X = CuArray(sample_inputs(N; seed = N))
    a, b = run!(eon, X), run!(eoff, X)
    synchronize()
    same = [Array(a[k]) == Array(b[k]) for k in 1:3]
    t_on = latency(() -> enqueue!(eon)).ms
    t_off = latency(() -> enqueue!(eoff)).ms
    @printf("   N = %5d   bitwise equal (signal, dT1, dT2) = %s   latency on %.2f ms, off %.2f ms\n",
            N, same, t_on, t_off)
end
close(eon); close(eoff)
