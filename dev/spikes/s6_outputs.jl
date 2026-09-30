# Spike S6 (2026-09-30): where can unwanted outputs go, and where can wanted
# ones sit?
#
# ① Can dT1/dT2 be bound to a null address when only the signal is wanted?
# ② If they must be real memory, can both share ONE scratch buffer (the
#   contents are discarded anyway), and does the signal stay bitwise equal?
# ③ The header demands 256-byte-aligned tensor addresses. A column view
#   out[:, j:k] of an (NTR, M) ComplexF32 matrix starts (j-1)·4800 bytes into
#   it, which is 256-aligned only when j-1 is a multiple of 4. Does TensorRT
#   reject, or silently accept, the misaligned ones — and are results right?
#
#     julia --project=dev dev/spikes/s6_outputs.jl

include("common.jl")
check_onnx()

e = Eng(engine_file("p_1_14400_40000"; min = 1, opt = 14400, max = 40000))
N = 14400
X = CuArray(sample_inputs(N))
ref = Array.(run!(e, X)); synchronize()
names = ("signal", "dT1", "dT2")

println("== ① null addresses for dT1/dT2")
S = CuMatrix{ComplexF32}(undef, NTR, N)
set_address!(e, "x", devptr(X)); set_address!(e, "signal", devptr(S))
ok = [set_address!(e, n, C_NULL) for n in ("dT1", "dT2")]
println("   setTensorAddress(null) accepted: ", ok)
r = enqueue!(e); synchronize()
println("   enqueueV3 → ", r, "   last TensorRT error: ", unsafe_string(ccall(sym(:sp_last_log), Cstring, ())))
println("   signal bitwise equal to the full run: ", Array(S) == ref[1])

println("\n== ② dT1 and dT2 sharing one scratch buffer")
fill!(S, 0)
scratch = CuMatrix{ComplexF32}(undef, NTR, N)
ok = [set_address!(e, n, devptr(scratch)) for n in ("dT1", "dT2")]
r = enqueue!(e); synchronize()
println("   bind accepted: ", ok, "   enqueueV3 → ", r)
println("   signal bitwise equal to the full run: ", Array(S) == ref[1])
println("   repeated 20×, signal still bitwise equal: ",
        all(1:20) do _; fill!(S, 0); enqueue!(e); synchronize(); Array(S) == ref[1]; end)

println("\n== ③ outputs as column views of a larger matrix, starting at column j")
M = N + 8
big = ntuple(_ -> CuMatrix{ComplexF32}(undef, NTR, M), 3)
for j in 1:6
    offset = (j - 1) * NTR * sizeof(ComplexF32)
    views = map(A -> view(A, :, j:j+N-1), big)
    foreach(A -> fill!(A, NaN32), big)
    ptrs = map(v -> reinterpret(Ptr{Cvoid}, pointer(v)), views)
    accepted = [set_address!(e, n, p) for (n, p) in zip(names, ptrs)]
    ran = all(accepted) ? enqueue!(e) : false
    synchronize()
    equal = ran && all(Array(views[k]) == ref[k] for k in 1:3)
    @printf("   j = %d  offset %5d B  (mod 256 = %3d)  bind accepted %-20s enqueue %-5s bitwise equal %s\n",
            j, offset, offset % 256, accepted, ran, equal)
end
println("   base pointer mod 256 of a fresh CuMatrix: ", UInt(pointer(big[1])) % 256)
close(e)

# ③b Chunked evaluation: the same 14400 voxels in pieces of different sizes
# (each piece its own call, own batch size) into column views of one matrix.
# Is every voxel's result independent of the batch it was computed in?
e = Eng(engine_file("p_1_14400_40000"; min = 1, opt = 14400, max = 40000))
println("\n== ③b chunked evaluation vs the whole block (N = $N)")
for cuts in ([7200], [4000, 8000, 12000], [5001], [1, 2, 3, 100, 9999])
    outs = ntuple(_ -> CuMatrix{ComplexF32}(undef, NTR, N), 3)
    bounds = [0; cuts; N]
    for (a, b) in zip(bounds[1:end-1] .+ 1, bounds[2:end])
        run!(e, X[:, a:b], map(A -> view(A, :, a:b), outs))
    end
    synchronize()
    println("   chunks ending at ", join(bounds[2:end], ", "), ": bitwise equal ",
            [Array(outs[k]) == ref[k] for k in 1:3])
end
close(e)
