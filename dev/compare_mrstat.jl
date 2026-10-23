# A comparison that ran one time (task 10, 2026-10-01): the engine of this
# package against the engine of MRSTAT.jl (models/model.engine), and both
# against EPG. On that date, both engines came from the same ONNX, and the
# voxels were the evaluation set of MRSTAT. dev/NOTES.md records the result.
# It is not a test, because it needs a checkout of MRSTAT.jl next to this one.
#
#     julia --project=scripts dev/compare_mrstat.jl
#
# The script reads ../MRSTAT.jl/models/model.engine and writes nothing. The
# engine of MRSTAT runs through the TensorRT layer of this package:
# libnb-trtinfer.so opens each engine of the installed TensorRT version.
#
# Since the fisp600-rank16 bundle, the two engines do not come from the same
# model, and the comparison is not correct. Before you run it again, build the
# upstream engine again and run the bootstrap.sh of MRSTAT again. The
# evaluation set now uses the domain of the card. Thus, it is the set of
# MRSTAT only for the domain of the 2026-10-01 card.

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
io = resolve_io(TensorRT.tensors(theirs), :signal_and_derivatives; model.ntr, model.rank)
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
println("$N voxels over the card's training domain; MRSTAT's engine: $MRSTAT_ENGINE")
println("   ", rpad("output", 8), rpad("ours = MRSTAT's", 17), rpad("max |ours - MRSTAT's|", 23), "NMAE ours / MRSTAT's vs EPG")
for k in (:signal, :∂T₁, :∂T₂)
    a, b = Array(pred[k]), Array(outs[k])
    @printf("   %-8s%-17s%-23.3g%.4f %% / %.4f %%\n", k, a == b ? "bitwise" : "no", maximum(abs, a .- b),
            100nmae(a, ref[k]), 100nmae(b, ref[k]))
end
close(theirs); close(ours)
