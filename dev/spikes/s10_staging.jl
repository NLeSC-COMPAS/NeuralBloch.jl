# Spike S10 (2026-10-01): how to stage the input without allocating.
# (Added after Phase 0, while implementing task 9; see dev/NOTES.md.)
#
# The engine reads one (3, N) Float32 matrix [T₁; T₂; B₁]; callers hand in
# three vectors. predict! must copy them into a staging buffer on the task's
# stream, and the plan requires that, after warm-up, predict! allocates
# nothing on the host. Do CUDA.jl's broadcasts (MRSTAT's three
# `X[k, :] .= v`) allocate host memory per call? Does one hand-written @cuda
# kernel? What does each cost on the GPU?
#
#     julia --project=dev dev/spikes/s10_staging.jl

using CUDACore, Printf, Statistics

N = 14400
X = CuMatrix{Float32}(undef, 3, 40000)
T₁, T₂, B₁ = CuVector{Float32}(rand(Float32, N)), CuVector{Float32}(rand(Float32, N)), CuVector{Float32}(rand(Float32, N))

broadcasts!(X, T₁, T₂, B₁, N) = (@views begin
    X[1, 1:N] .= T₁
    X[2, 1:N] .= T₂
    X[3, 1:N] .= B₁
end; nothing)

function stage_kernel!(X, T₁, T₂, B₁, N)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if j <= N
        @inbounds X[1, j] = T₁[j]
        @inbounds X[2, j] = T₂[j]
        @inbounds X[3, j] = B₁[j]
    end
    return nothing
end
kernel!(X, T₁, T₂, B₁, N) = (@cuda threads = 256 blocks = cld(N, 256) stage_kernel!(X, T₁, T₂, B₁, Int32(N)); nothing)

function measure(f, name)
    f(X, T₁, T₂, B₁, N); synchronize()
    bytes = minimum(@allocated(f(X, T₁, T₂, B₁, N)) for _ in 1:20)
    synchronize()
    ts = map(1:200) do _
        synchronize()
        t0 = time_ns(); f(X, T₁, T₂, B₁, N); synchronize(); (time_ns() - t0) / 1e3
    end
    host = median(map(1:200) do _
        synchronize(); t0 = time_ns(); f(X, T₁, T₂, B₁, N); dt = (time_ns() - t0) / 1e3; synchronize(); dt
    end)
    @printf("%-28s host bytes/call %5d   wall %6.1f µs (median)   host submit %5.1f µs\n", name, bytes, median(ts), host)
    return Array(X[:, 1:N])
end

println("N = $N, CUDACore ", pkgversion(CUDACore))
a = measure(broadcasts!, "3 broadcasts (MRSTAT)")
b = measure(kernel!, "1 @cuda kernel")
println("same staged values: ", a == b == vcat(Array(T₁)', Array(T₂)', Array(B₁)'))

# Strided device-to-device copies instead of a kernel: row k of X is every
# third Float32, so T[j] goes to byte 12(j-1) + 4(k-1) (pitch 12), from a
# contiguous source (pitch 4). CUDACore.unsafe_copy2d! wraps cuMemcpy2DAsync.
function copies!(X, T₁, T₂, B₁, N)
    s = CUDACore.stream()
    for (k, v) in ((1, T₁), (2, T₂), (3, B₁))
        CUDACore.unsafe_copy2d!(pointer(X, k), CUDACore.DeviceMemory, pointer(v), CUDACore.DeviceMemory,
                                1, N; dstPitch = 12, srcPitch = 4, async = true, stream = s)
    end
    return nothing
end
c = measure(copies!, "3 cuMemcpy2DAsync")
println("same staged values: ", c == a)

println("\n== scaling (wall µs, median of 200)")
for n in (1000, 14400, 40000)
    global N = n
    global T₁, T₂, B₁ = (CuVector{Float32}(rand(Float32, n)) for _ in 1:3)
    ts = map((kernel!, copies!)) do f
        f(X, T₁, T₂, B₁, n); synchronize()
        median(map(1:200) do _
            synchronize(); t0 = time_ns(); f(X, T₁, T₂, B₁, n); synchronize(); (time_ns() - t0) / 1e3
        end)
    end
    @printf("   N = %5d   @cuda kernel %6.1f µs   3 cuMemcpy2DAsync %6.1f µs\n", n, ts...)
end
