# One-time comparison (task 10, 2026-10-01): this package's engine against
# MRSTAT.jl's existing one (models/model.engine, built upstream from the same
# ONNX with upstream's settings), on MRSTAT's evaluation set, and both
# against EPG. The result is recorded in dev/NOTES.md; it is deliberately not
# a test, since it needs a sibling checkout.
#
#     julia --project=scripts dev/compare_mrstat.jl
#
# Reads ../MRSTAT.jl/models/model.engine and writes nothing. MRSTAT's engine
# runs through this package's own TensorRT layer: libnb-trtinfer.so opens any
# engine of the installed TensorRT version.

using NeuralBloch
using NeuralBloch: TensorRT, resolve_io
using CUDACore: CuArray, CuMatrix, synchronize
using Printf

include(joinpath(@__DIR__, "..", "scripts", "reference.jl"))

const MRSTAT_ENGINE = normpath(joinpath(@__DIR__, "..", "..", "MRSTAT.jl", "models", "model.engine"))

model = Model()
v = evaluation_set(model)
N = length(v.T₁)

ours = Engine(model)
pred = predict(ours, gpu_voxels(v))

theirs = TensorRT.Handle(MRSTAT_ENGINE)
io = resolve_io(TensorRT.tensors(theirs), model.ntr)
X = CuArray(Float32.(permutedims(hcat(v.T₁, v.T₂, v.B₁))))       # (3, N): the engine's x
outs = (signal = CuMatrix{ComplexF32}(undef, model.ntr, N), ∂T₁ = CuMatrix{ComplexF32}(undef, model.ntr, N),
        ∂T₂ = CuMatrix{ComplexF32}(undef, model.ntr, N))
TensorRT.set_shape!(theirs, io.x, (N, 3))
TensorRT.set_address!(theirs, io.x, reinterpret(Ptr{Cvoid}, pointer(X)))
for k in (:signal, :∂T₁, :∂T₂)
    TensorRT.set_address!(theirs, io[k], reinterpret(Ptr{Cvoid}, pointer(outs[k])))
end
TensorRT.run!(theirs)
synchronize()

ref = epg(model, v)
println("$N voxels (MRSTAT.jl's evaluation set); MRSTAT's engine: $MRSTAT_ENGINE")
println("   ", rpad("output", 8), rpad("ours = MRSTAT's", 17), rpad("max |ours - MRSTAT's|", 23), "NMAE ours / MRSTAT's vs EPG")
for k in (:signal, :∂T₁, :∂T₂)
    a, b = Array(pred[k]), Array(outs[k])
    @printf("   %-8s%-17s%-23.3g%.4f %% / %.4f %%\n", k, a == b ? "bitwise" : "no", maximum(abs, a .- b),
            100nmae(a, ref[k]), 100nmae(b, ref[k]))
end
close(theirs); close(ours)
