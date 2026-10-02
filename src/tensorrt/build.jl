# Building engines: nb-trt-build runs as a subprocess, its product goes
# into the cache under the engine's key (cache.jl).

const ENGINE_FILE = "model.engine"

"""
    BuildSettings(; batch = (1, 14400, 40000), tf32 = false, optlevel = 3)

How an engine is built:

- `batch = (min, opt, max)`: the batch sizes the engine accepts, and the one
  its kernels are tuned for. The default serves every batch up to 40000 at
  upstream's speed and scratch memory (≈11 KiB per voxel of `max`); see
  dev/NOTES.md, S4.
- `tf32`: allow TF32 inner products. Off by default: accuracy first; on
  GPUs before Ampere it changes nothing (S3).
- `optlevel`: TensorRT's builder optimization level, 0–5.

The settings are part of the engine's cache key.
"""
struct BuildSettings
    batch::NTuple{3,Int}
    tf32::Bool
    optlevel::Int

    function BuildSettings(; batch = (1, 14400, 40000), tf32::Bool = false, optlevel::Integer = 3)
        lo, opt, hi = batch
        1 ≤ lo ≤ opt ≤ hi || throw(ArgumentError(
            "batch = $(Tuple(batch)) must be (min, opt, max) with 1 ≤ min ≤ opt ≤ max"))
        0 ≤ optlevel ≤ 5 || throw(ArgumentError("optlevel = $optlevel must be in 0:5"))
        return new((Int(lo), Int(opt), Int(hi)), tf32, Int(optlevel))
    end
end

settings_dict(s::BuildSettings) =
    Dict{String,Any}("batch" => collect(s.batch), "tf32" => s.tf32, "optlevel" => s.optlevel)

"""
    BuildError(msg)

Building an engine failed; the message carries the builder's own output.
"""
struct BuildError <: Exception
    msg::String
end

Base.showerror(io::IO, e::BuildError) = print(io, "BuildError: ", e.msg)

# What each exit code of nb-trt-build means (see builder.cpp).
const BUILDER_EXITS = Dict(
    1 => "the builder rejected its arguments (a bug in NeuralBloch)",
    2 => "the ONNX parser rejected the model",
    3 => "the ONNX does not describe a network NeuralBloch can build",
    4 => "TensorRT could not build the engine",
    5 => "the engine could not be written",
    6 => "TensorRT could not start on the GPU",
    7 => "the builder hit an unexpected C++ exception",
)

"""
    run_builder(native, onnx, out, settings, gpu; timeout = 600) -> Dict

Run `native.builder` on `onnx`, writing the engine to `out`, on `gpu` alone;
return the builder's report (what it applied and built). Throws a
[`BuildError`](@ref) with the builder's messages when it fails, and when it
does not finish within `timeout` seconds (a hang guard: builds take seconds).
"""
function run_builder(native::Native, onnx::AbstractString, out::AbstractString,
                     settings::BuildSettings, gpu::GPU; timeout::Real = 600)
    cmd = `$(native.builder) --onnx $onnx --output $out --batch $(join(settings.batch, ','))
           --tf32 $(Int(settings.tf32)) --optlevel $(settings.optlevel)`
    # The GPU by UUID: absolute, whatever CUDA_VISIBLE_DEVICES this process has (S7).
    env = merge(ENV, Dict("CUDA_VISIBLE_DEVICES" => "GPU-$(gpu.uuid)"))
    report, log = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(setenv(cmd, env)); stdout = report, stderr = log); wait = false)
    timedout = Ref(false)
    watchdog = Timer(timeout) do _
        timedout[] = process_running(p)
        timedout[] && kill(p, Base.SIGKILL)
    end
    try
        wait(p)
    catch                                       # an interrupt must not leave the builder running
        kill(p, Base.SIGKILL)
        rethrow()
    finally
        close(watchdog)
    end
    success(p) && return TOML.parse(String(take!(report)))

    why = timedout[] ? "the builder did not finish within $timeout s and was stopped" :
          p.termsignal != 0 ? "the builder died of signal $(p.termsignal)" :
          get(BUILDER_EXITS, p.exitcode, "the builder failed with exit code $(p.exitcode)")
    msg = "building a TensorRT engine from $onnx for $(gpu.name) failed: $why. The builder said:\n" *
          rstrip(String(take!(log)))
    # TensorRT does not say "out of memory" when that is the cause (S7).
    if p.exitcode in (4, 6) || p.termsignal != 0
        msg *= "\n(free GPU memory: $(round(CUDACore.free_memory() / 2^30; digits = 2)) GiB; " *
               "a build needs about 0.5 GiB)"
    end
    throw(BuildError(msg))
end

"""
    build(onnx; settings = BuildSettings(), sys = detect(), root) -> String

The path of a TensorRT engine built from `onnx` for the calling task's GPU:
from the cache (`root`, by default in [`cache_root`](@ref)) when one was
built before for the same ONNX content, TensorRT, GPU model, settings and C++
sources; otherwise built now, announced with one line (it takes seconds).

Building compiles the C++ first when needed ([`compile`](@ref)), returns
the memory CUDA.jl's pool holds to the driver (the builder is another process
and would not see it otherwise), and runs `nb-trt-build` as a
subprocess. Throws a [`PrerequisiteError`](@ref) when the system lacks
something, a [`BuildError`](@ref) when the build fails.
"""
function build(onnx::AbstractString; settings::BuildSettings = BuildSettings(), sys::System = detect(),
               root::AbstractString = joinpath(cache_root(), "engines"))
    require(sys)
    native = compile(sys)
    key = engine_key(; onnx = bytes2hex(open(SHA.sha256, onnx)), tensorrt = sys.nvinfer.version,
                     gpu = sys.gpu.name, capability = sys.gpu.capability,
                     settings = settings_dict(settings), csrc = native.csrc)
    dir = getorbuild(root, key; what = "a TensorRT engine for $(sys.gpu.name)") do tmp
        @info "Building a TensorRT engine for $(sys.gpu.name) with batch sizes $(join(settings.batch, '/')) " *
              "(min/opt/max): once per GPU model, TensorRT version and settings; takes a few seconds."
        CUDACore.reclaim()
        return run_builder(native, onnx, joinpath(tmp, ENGINE_FILE), settings, sys.gpu)
    end
    return joinpath(dir, ENGINE_FILE)
end
