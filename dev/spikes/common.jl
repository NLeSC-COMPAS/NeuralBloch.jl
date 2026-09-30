# Shared helpers for the Phase 0 spikes (2026-09-30). Throwaway code: it is
# part of neither the package nor the test suite. Results: dev/NOTES.md.
#
# Every spike runs in the dev/ environment (CUDACore only), from the
# repository root:
#
#     julia --project=dev dev/spikes/<spike>.jl
#
# Products (compiled helpers, engines) go to .work/spikes/, which is gitignored.
# The spikes read two files from sibling checkouts, never write to them:
#   ../NeuralBloch/outputs/model.onnx   upstream ONNX (tensorRT branch, 9f4db84)
#   ../MRSTAT.jl/models/model.engine    MRSTAT's engine, built upstream from it

using CUDACore, Libdl, Printf, Random, SHA, Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const WORK = joinpath(ROOT, ".work", "spikes")
const ONNX = normpath(joinpath(ROOT, "..", "NeuralBloch", "outputs", "model.onnx"))
const ONNX_SHA = "3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607"
const REF_ENGINE = normpath(joinpath(ROOT, "..", "MRSTAT.jl", "models", "model.engine"))
const CUDA_INC = "/usr/local/cuda/include"      # cuda_runtime_api.h (cuda-cudart-dev)
const NTR = 600

mkpath(WORK)

check_onnx() = bytes2hex(open(sha256, ONNX)) == ONNX_SHA ||
    error("$ONNX is not the ONNX the plan pins (sha256 $ONNX_SHA)")

# g++ with the deb headers only: /usr/include/x86_64-linux-gnu is on g++'s
# default search path, so TensorRT needs no -I; only CUDA's headers do.
function compile(src, out; shared::Bool, libs)
    kind = shared ? ["-fPIC", "-shared"] : String[]
    run(`g++ -O2 -std=c++17 -Wall -isystem $CUDA_INC $kind $(joinpath(@__DIR__, src)) -o $out
         $(["-l$l" for l in libs]) -Wl,--no-undefined`)
    return out
end

const ENGINE_LIB = joinpath(WORK, "libspike_engine.so")
const BUILDER = joinpath(WORK, "spike_builder")
compile_engine_lib() = compile("spike_engine.cpp", ENGINE_LIB; shared = true, libs = ["nvinfer"])
compile_builder() = compile("spike_builder.cpp", BUILDER; shared = false, libs = ["nvinfer", "nvonnxparser"])

# Run the spike builder; returns (key => value) pairs it printed and the wall time.
function build(out; min, opt, max, tf32 = -1, level = -1, workspace_mib = 0, env = nothing)
    isfile(BUILDER) || compile_builder()
    cmd = `$BUILDER $ONNX $out $min $opt $max $tf32 $level $workspace_mib`
    env === nothing || (cmd = setenv(cmd, env))
    t = @elapsed (text = read(cmd, String))
    kv = Dict(strip(k) => strip(v) for (k, v) in (split(l, " = ") for l in split(chomp(text), '\n')))
    return kv, t
end

# Build into .work/spikes/<name>.engine unless that file already exists.
function engine_file(name; kw...)
    path = joinpath(WORK, "$name.engine")
    if !isfile(path)
        kv, t = build(path; kw...)
        @printf("   built %s in %.1f s (builder: %s s; tf32 %s, level %s, workspace %s)\n",
                name, t, kv["build_s"], kv["tf32"], kv["optlevel"], kv["workspace"])
    end
    return path
end

filesha(path) = bytes2hex(open(sha256, path))

# --- the spike engine library, via ccall --------------------------------------

const LIB = Ref{Ptr{Cvoid}}(C_NULL)
function lib()
    if LIB[] == C_NULL
        isfile(ENGINE_LIB) || compile_engine_lib()
        LIB[] = dlopen(ENGINE_LIB)
    end
    return LIB[]
end
sym(name) = dlsym(lib(), name)

struct Eng
    h::Ptr{Cvoid}
end
function Eng(path::AbstractString)
    CUDACore.context()                       # TensorRT must see CUDA.jl's (primary) context
    h = ccall(sym(:sp_open), Ptr{Cvoid}, (Cstring,), path)
    h == C_NULL && error("sp_open($path): ", unsafe_string(ccall(sym(:sp_last_log), Cstring, ())))
    return Eng(h)
end
Base.close(e::Eng) = ccall(sym(:sp_close), Cvoid, (Ptr{Cvoid},), e.h)
device_memory(e::Eng) = ccall(sym(:sp_device_memory), Int64, (Ptr{Cvoid},), e.h)
function profile(e::Eng, name = "x")
    dims = zeros(Int64, 8)
    map(0:2) do sel
        ccall(sym(:sp_profile), Int32, (Ptr{Cvoid}, Cstring, Int32, Ptr{Int64}), e.h, name, sel, dims)
        Int(dims[1])
    end
end
set_shape!(e::Eng, N) = ccall(sym(:sp_set_shape), Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Int32),
                              e.h, "x", Int64[N, 3], 2) == 1 || error("setInputShape($N) rejected")
set_address!(e::Eng, name, p) = ccall(sym(:sp_set_address), Int32, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}),
                                      e.h, name, p) == 1
devptr(A::CuArray) = reinterpret(Ptr{Cvoid}, pointer(A))
streamptr() = reinterpret(Ptr{Cvoid}, CUDACore.stream().handle)
enqueue!(e::Eng) = ccall(sym(:sp_enqueue), Int32, (Ptr{Cvoid}, Ptr{Cvoid}), e.h, streamptr()) == 1

# Bind x and the three outputs and run once; returns the outputs (on the task stream).
function run!(e::Eng, X::AbstractMatrix{Float32}, outs = ntuple(_ -> CuMatrix{ComplexF32}(undef, NTR, size(X, 2)), 3))
    set_shape!(e, size(X, 2))
    set_address!(e, "x", devptr(X)) || error("bind x")
    for (name, A) in zip(("signal", "dT1", "dT2"), outs)
        set_address!(e, name, devptr(A)) || error("bind $name")
    end
    enqueue!(e) || error("enqueueV3 failed")
    return outs
end

# --- inputs from the training distribution -----------------------------------

# T₁ ∈ [0.1, 5] s and T₂ ∈ [0.01, 2] s log-uniform, B₁ ∈ [0.8, 1.2] uniform,
# keeping only T₂ < T₁ (data/data_generation.jl:78) by rejection.
function sample_inputs(N; seed = 1)
    rng = Xoshiro(seed)
    logu(lo, hi) = exp(log(lo) + rand(rng) * (log(hi) - log(lo)))
    X = Matrix{Float32}(undef, 3, N)
    for j in 1:N
        while true
            T₁, T₂ = logu(0.1, 5.0), logu(0.01, 2.0)
            if T₂ < T₁
                X[:, j] .= (T₁, T₂, 0.8 + 0.4rand(rng))
                break
            end
        end
    end
    return X
end

# Largest |a - b| relative to the largest |b|, over the whole output, and the
# worst voxel's max |a - b| relative to that voxel's max |b|.
function relerr(a::AbstractMatrix, b::AbstractMatrix)
    a, b = Array(a), Array(b)
    whole = maximum(abs.(a .- b)) / maximum(abs.(b))
    pervoxel = maximum(maximum(abs.(a[:, j] .- b[:, j])) / maximum(abs.(b[:, j])) for j in axes(b, 2))
    return (; whole, pervoxel)
end

# Median wall time of f() with a device sync after each call, and the median
# host time of the submission alone (queue drained first).
function latency(f; n = 50)
    f(); synchronize()
    ts = map(1:n) do _
        synchronize()
        1e3 * @elapsed (f(); synchronize())
    end
    hs = map(1:n) do _
        synchronize()
        t0 = time_ns(); f(); dt = (time_ns() - t0) / 1e3
        synchronize()
        dt
    end
    return (ms = median(ts), host_us = median(hs))
end
