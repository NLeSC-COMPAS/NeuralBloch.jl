# Spike S1 (2026-09-30): headers, linking and version detection.
#
# ① Do the deb packages' headers suffice to compile (TensorRT's -dev packages
#   plus cuda_runtime_api.h from cuda-cudart-dev)?
# ② Without a CUDA graph, does the runtime library need libcudart at all?
# ③ Can Julia ccall getInferLibVersion / getNvOnnxParserVersion directly?
#
#     julia --project=dev dev/spikes/s1_link.jl

include("common.jl")

println("== ① compile with the deb headers only (+ -isystem $CUDA_INC)")
compile_engine_lib(); compile_builder()
println("   ok: ", ENGINE_LIB, "\n   ok: ", BUILDER)

println("\n== which headers were actually used (g++ -H), outside the C++ stdlib")
hdrs = read(pipeline(`g++ -std=c++17 -H -fsyntax-only -isystem $CUDA_INC $(joinpath(@__DIR__, "spike_builder.cpp"))`;
                     stderr = pipeline(`grep -oE '/[^ ]+'`)), String)
used = unique(filter(h -> !occursin("/c++/", h) && !startswith(h, "/usr/include/x86_64-linux-gnu/bits") &&
                          !startswith(h, "/usr/include/x86_64-linux-gnu/sys") &&
                          !startswith(h, "/usr/include/x86_64-linux-gnu/gnu"), split(hdrs)))
dirs = unique(dirname.(used))
for d in dirs
    n = count(h -> dirname(h) == d, used)
    println("   ", rpad(d, 60), n, " headers")
end
cudahdrs = filter(h -> startswith(h, realpath(CUDA_INC)) || startswith(h, CUDA_INC), used)
owners = unique(map(cudahdrs) do h
    strip(first(split(read(`dpkg -S $(realpath(h))`, String), ':')))
end)
println("   CUDA headers come from: ", join(owners, ", "))

println("\n== ② undefined cuda* symbols (nm -u) and dependencies (ldd)")
for f in (ENGINE_LIB, BUILDER)
    und = filter(s -> occursin(r"^cuda|^cu[A-Z]", s), split(read(`nm -u --demangle=none $f`, String)))
    println("   ", basename(f), ": undefined cuda* symbols = ", isempty(und) ? "none" : join(und, " "))
    deps = [first(split(strip(l))) for l in split(read(`ldd $f`, String), '\n') if !isempty(strip(l))]
    println("     ldd: ", join(deps, " "))
    occursin("not found", read(`ldd $f`, String)) && println("     !! missing libraries")
end

println("\n== ③ versions via ccall, straight from the system libraries")
nvinfer = dlopen("libnvinfer.so.11")
parser = dlopen("libnvonnxparser.so.11")
v = ccall(dlsym(nvinfer, :getInferLibVersion), Int32, ())
pv = ccall(dlsym(parser, :getNvOnnxParserVersion), Int32, ())
println("   getInferLibVersion()     = $v  → $(v ÷ 10000).$(v ÷ 100 % 100).$(v % 100)")
println("   getNvOnnxParserVersion() = $pv  → $(pv ÷ 10000).$(pv ÷ 100 % 100).$(pv % 100)")
println("   dlpath: ", dlpath(nvinfer), " | ", dlpath(parser))
println("   spike lib: header $(ccall(sym(:sp_header_version), Int32, ())), runtime $(ccall(sym(:sp_lib_version), Int32, ()))")

println("\n== run the spike library once, on MRSTAT's engine")
e = Eng(REF_ENGINE)
println("   profile x (MIN/OPT/MAX batch) = ", profile(e), "   scratch = ",
        round(device_memory(e) / 2^20; digits = 1), " MiB")
X = CuArray(sample_inputs(14400))
s, d1, d2 = run!(e, X)
synchronize()
for (n, A) in (("signal", s), ("dT1", d1), ("dT2", d2))
    a = Array(A)
    println("   ", rpad(n, 6), " finite = ", all(isfinite, a), "   max |.| = ", maximum(abs, a))
end
close(e)
