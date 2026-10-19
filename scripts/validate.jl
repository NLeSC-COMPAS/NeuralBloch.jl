# Validation of NeuralBloch.jl on this machine: a report, not a test.
#
#     julia --project=scripts scripts/validate.jl
#
# Three checks (plan.md, "Definition of done"), each reported PASS or FAIL;
# the exit code is 1 if any fails:
#
#   accuracy  NMAE against the ground truth: the FISP3D EPG simulation of
#             BlochSimulators (the simulation that made the training data),
#             with the derivative method and the training sequence of the
#             model card. The evaluation set is in the training domain of
#             the card (14400 points, seed 20260826).
#   memory    200 sequential predict! calls for each target (signal and
#             derivatives, signal, coefficients): no host or GPU allocation,
#             no GC.
#   speed     on par with MRSTAT.jl's engine: 1.30 ms for 14400 voxels
#             (plan.md, appendix A), within 10 %.
#
# Timings are medians of 50 calls with a device synchronization after each,
# appendix A's method, so the numbers compare directly.

using NeuralBloch
using NeuralBloch: TARGETS
using CUDACore: CUDACore, synchronize
using Printf, Statistics

include("reference.jl")

# The NMAE thresholds of each model, by family and version: for each output,
# 1.05 × the value that this set measured, rounded up (dev/NOTES.md). A
# different build of the same ONNX changes the sixth digit of the NMAE. A real
# regression changes the second digit. The thresholds of one model do not
# apply to another: for a model without thresholds, the script gives the NMAE
# and FAIL.
const THRESHOLDS = Dict(
    # Measured 2026-10-15 ("The fisp600-rank16 v1 bundle"), the same network as 0.1.0-beta.1
    ("fisp600-rank16", v"0.1.0-beta.1") => (signal = 0.0128, ∂T₁ = 0.0074, ∂T₂ = 0.0121),
)
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
thresholds = get(THRESHOLDS, (model.family, model.version), nothing)
for k in (:signal, :∂T₁, :∂T₂)
    e = nmae(pred[k], ref[k])
    if thresholds === nothing
        @printf("   %-7s NMAE %.4f %%   (no threshold)   %s\n", k, 100e, verdict!("accuracy $k", false))
    else
        @printf("   %-7s NMAE %.4f %%   (threshold %.2f %%)   %s\n", k, 100e, 100thresholds[k],
                verdict!("accuracy $k", e ≤ thresholds[k]))
    end
end
thresholds === nothing && println("   There are no thresholds for $(model.family) v$(model.version). ",
                                  "Measure the model, examine the NMAE, and add them to THRESHOLDS.")

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
for target in TARGETS
    host, device, gcs = allocations(outputs(engine, N; target), engine, p)
    @printf("   %-23s host %d bytes, GPU %d bytes, %d GCs   %s\n", target, host, device, gcs,
            verdict!("memory, $target", host == 0 && device == 0 && gcs == 0))
end

# --- speed ------------------------------------------------------------------------------

println("\n== speed: median ms per call (device synchronized after each)")
println("   ", rpad("N", 8), rpad("predict!", 11), rpad("signal", 9), rpad("coefficients", 14),
        rpad("engine alone", 14), rpad("host µs", 10), "MRSTAT's predict! (appendix A: graph replay)")
appendix_a = Dict(10000 => 1.03, 14400 => 1.30, 40000 => 3.43)
for n in (1000, 10000, 14400, 40000)
    q = gpu_voxels(evaluation_set(model, n; seed = n))
    outs = NamedTuple{TARGETS}(map(target -> outputs(engine, n; target), TARGETS))
    ms = map(out -> latency(() -> predict!(out, engine, q)), outs)      # columns 2 to 4, in order
    full = outs.signal_and_derivatives
    submit = host_submission(() -> predict!(full, engine, q))
    predict!(full, engine, q)                                  # binds this batch; then the run alone
    t_engine = latency(() -> NeuralBloch.TensorRT.run!(engine.targets.signal_and_derivatives.handle))
    @printf("   %-8d%-11.3f%-9.3f%-14.3f%-14.3f%-10.1f%s\n", n, values(ms)..., t_engine, submit,
            haskey(appendix_a, n) ? @sprintf("%.2f", appendix_a[n]) : "—")
    if n == 14400
        @printf("   at 14400 voxels: %.3f ms against a target of %.2f ms   %s\n", ms.signal_and_derivatives,
                LATENCY_TARGET_MS, verdict!("speed", ms.signal_and_derivatives ≤ LATENCY_TARGET_MS))
    end
end

# --- verdict ----------------------------------------------------------------------------

close(engine)
failed = [name for (name, ok) in verdicts if !ok]
println("\n", isempty(failed) ? "All $(length(verdicts)) checks passed." :
                               "FAILED: " * join(failed, ", "))
exit(isempty(failed) ? 0 : 1)
