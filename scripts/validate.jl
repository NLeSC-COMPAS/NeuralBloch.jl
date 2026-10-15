# Validation of NeuralBloch.jl on this machine: a report, not a test.
#
#     julia --project=scripts scripts/validate.jl
#
# Three targets (plan.md, "Definition of done"), each reported PASS or FAIL;
# the exit code is 1 if any fails:
#
#   accuracy  NMAE against ground truth, i.e. BlochSimulators' FISP3D EPG
#             simulation (the generator of the training data) with the
#             derivative method the model card states, on the training
#             sequence as the card describes it. The evaluation set covers
#             the card's training domain (14400 points, seed 20260826).
#   memory    200 consecutive predict! calls: no host or GPU allocation, no GC.
#   speed     on par with MRSTAT.jl's engine: 1.30 ms for 14400 voxels
#             (plan.md, appendix A), within 10 %.
#
# Timings are medians of 50 calls with a device synchronization after each,
# appendix A's method, so the numbers compare directly.

using NeuralBloch
using CUDACore: CUDACore, synchronize
using Printf, Statistics

include("reference.jl")

# NMAE thresholds per output: 1.05 × what this set measured for
# fisp600-rank16 v1 on 2026-10-15, rounded up (dev/NOTES.md). A different
# build of the same ONNX moves NMAE in the sixth digit; a real regression
# moves the second.
const THRESHOLDS = (signal = 0.0128, ∂T₁ = 0.0074, ∂T₂ = 0.0121)
const LATENCY_TARGET_MS = 1.30 * 1.10                          # at 14400 voxels

function latency(f; n = 50)
    f(); synchronize()
    return median(map(_ -> (synchronize(); 1e3 * @elapsed (f(); synchronize())), 1:n))
end

function host_submission(f; n = 50)
    f(); synchronize()
    return median(map(1:n) do _
        synchronize()
        t = @elapsed f()
        synchronize()
        1e6t
    end)
end

verdicts = Pair{String,Bool}[]
verdict!(name, ok) = (push!(verdicts, name => ok); ok ? "PASS" : "FAIL")

NeuralBloch.versioninfo()
model = Model()
engine = Engine(model)
println("\nengine: ", engine)

# --- accuracy ------------------------------------------------------------------------

println("\n== accuracy: NMAE against BlochSimulators $(pkgversion(BlochSimulators)) EPG + ",
        model.derivatives.method)
v = evaluation_set(model)
N = length(v.T₁)
p = gpu_voxels(v)
checkdomain(engine, p)
pred = predict(engine, p)
t = @elapsed ref = epg(model, v)
@printf("   %d voxels over the card's training domain; EPG reference in %.1f s\n", N, t)
for k in (:signal, :∂T₁, :∂T₂)
    e = nmae(pred[k], ref[k])
    @printf("   %-7s NMAE %.4f %%   (threshold %.2f %%)   %s\n", k, 100e, 100THRESHOLDS[k],
            verdict!("accuracy $k", e ≤ THRESHOLDS[k]))
end

# --- memory ----------------------------------------------------------------------------

println("\n== memory: 200 consecutive predict! calls, $N voxels")
# In a function, on local variables: at top level every access to a global
# would allocate a box of its own.
function allocations(out, engine, p; calls = 100)
    predict!(out, engine, p); predict!(out, engine, p)
    synchronize()
    GC.gc()
    gc0 = Base.gc_num()
    host = @allocated for _ in 1:calls
        predict!(out, engine, p)
    end
    device = CUDACore.@allocated for _ in 1:calls
        predict!(out, engine, p)
    end
    synchronize()
    return host, device, Base.GC_Diff(Base.gc_num(), gc0).pause
end
for (name, out) in (("signal + derivatives", outputs(engine, N)), ("signal only", outputs(engine, N; derivatives = false)))
    host, device, gcs = allocations(out, engine, p)
    @printf("   %-21s host %d bytes, GPU %d bytes, %d GCs   %s\n", name, host, device, gcs,
            verdict!("memory, $name", host == 0 && device == 0 && gcs == 0))
end

# --- speed ------------------------------------------------------------------------------

println("\n== speed: median ms per call (device synchronized after each)")
println("   ", rpad("N", 8), rpad("predict!", 11), rpad("signal only", 13), rpad("engine alone", 14),
        rpad("host µs", 10), "MRSTAT's predict! (appendix A: graph replay)")
appendix_a = Dict(10000 => 1.03, 14400 => 1.30, 40000 => 3.43)
for n in (1000, 10000, 14400, 40000)
    q = gpu_voxels(evaluation_set(model, n; seed = n))
    full, signal = outputs(engine, n), outputs(engine, n; derivatives = false)
    t_full = latency(() -> predict!(full, engine, q))
    t_signal = latency(() -> predict!(signal, engine, q))
    submit = host_submission(() -> predict!(full, engine, q))
    predict!(full, engine, q)                                  # binds this batch; then the run alone
    t_engine = latency(() -> NeuralBloch.TensorRT.run!(engine.handle))
    @printf("   %-8d%-11.3f%-13.3f%-14.3f%-10.1f%s\n", n, t_full, t_signal, t_engine, submit,
            haskey(appendix_a, n) ? @sprintf("%.2f", appendix_a[n]) : "—")
    if n == 14400
        @printf("   at 14400 voxels: %.3f ms against a target of %.2f ms   %s\n", t_full, LATENCY_TARGET_MS,
                verdict!("speed", t_full ≤ LATENCY_TARGET_MS))
    end
end

# --- verdict ----------------------------------------------------------------------------

close(engine)
failed = [name for (name, ok) in verdicts if !ok]
println("\n", isempty(failed) ? "All $(length(verdicts)) targets met." :
                               "FAILED: " * join(failed, ", "))
exit(isempty(failed) ? 0 : 1)
