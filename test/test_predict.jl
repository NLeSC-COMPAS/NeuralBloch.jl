# predict.jl: Engine, outputs, predict!, predict, checkdomain and the
# metadata queries. The T2 testsets follow the plan's predict! checklist.

using NeuralBloch: Engine, Model, outputs, predict!, predict, checkdomain, ntr, domain, batchprofile, sequence
using CUDACore: CUDACore, CuArray, CuMatrix, CuVector, synchronize
using StructArrays: StructArray

# Voxels inside the training domain (T₂ < T₁), reproducibly, on the CPU.
function voxels_cpu(N; seed = 1)
    u(k, j) = mod(seed * (0.618034, 0.414214, 0.732051)[k] * j + 0.1k, 1)
    T₁ = [Float32(exp(log(0.1) + u(1, j) * log(50))) for j in 1:N]
    T₂ = [Float32(min(exp(log(0.01) + u(2, j) * log(200)), 0.9 * T₁[j])) for j in 1:N]
    B₁ = [Float32(0.8 + 0.4u(3, j)) for j in 1:N]
    return (; T₁, T₂, B₁)
end
voxels(N; seed = 1) = map(CuArray, voxels_cpu(N; seed))

const ENGINES = Dict{NTuple{3,Int},Any}()
engine(; batch = (1, 14400, 40000)) = get!(() -> Engine(; batch), ENGINES, batch)

sentinel(dims...) = fill!(CuMatrix{ComplexF32}(undef, dims...), ComplexF32(NaN))
untouched(A) = (synchronize(); all(isnan, Array(A)))
relative(a, b) = maximum(abs, Array(a) .- Array(b)) / maximum(abs, Array(b))

tier(:T0, "predict: model metadata and checkdomain") do
    m = Model()
    @test ntr(m) == 600
    @test domain(m) == (T₁ = (0.1, 5.0), T₂ = (0.01, 2.0), B₁ = (0.8, 1.2))
    s = sequence(m)
    @test s.kind == "FISP3D" && s.TR == 0.00558 && length(s.rf_angle) == 600

    p = voxels_cpu(100)
    @test checkdomain(m, p) === p                               # works on CPU vectors too
    @test checkdomain(m, (; B₁ = p.B₁)) isa NamedTuple          # any subset of the properties
    for (name, bad, words) in ((:T₁, 5.5f0, "p.T₁ ranges over"), (:T₁, 0.05f0, "[0.1, 5.0]"),
                               (:T₂, 2.5f0, "p.T₂"), (:B₁, 1.25f0, "[0.8, 1.2]"), (:B₁, NaN32, "p.B₁"))
        q = map(copy, p)
        getproperty(q, name)[7] = bad
        if name === :T₂                                         # keep T₂ < T₁ out of the picture
            q.T₁[7] = 4.9f0
        end
        err = @test_throws ArgumentError checkdomain(m, q)
        @test occursin(words, sprint(showerror, err.value))
        @test occursin("training domain", sprint(showerror, err.value))
    end
    q = map(copy, p)
    q.T₁[3], q.T₂[3] = 0.5f0, 0.5f0                              # T₂ = T₁, both in range
    q.T₁[9], q.T₂[9] = 0.3f0, 0.6f0                              # T₂ > T₁, both in range
    err = @test_throws ArgumentError checkdomain(m, q)
    @test occursin("2 voxels have T₂ ≥ T₁", sprint(showerror, err.value))
    @test_throws ArgumentError checkdomain(m, (; ρ = p.B₁))
end

tier(:T2, "predict: engine metadata") do
    e = engine()
    @test ntr(e) == 600
    @test batchprofile(e) == (min = 1, opt = 14400, max = 40000)
    @test domain(e) == domain(Model())
    @test sequence(e) == sequence(Model())
    @test isopen(e)
    @test occursin("NTR 600", sprint(show, e))
    p = voxels(10)
    @test checkdomain(e, p) === p                               # and on the GPU
end

tier(:T2, "predict: predict and predict! agree, repeatedly") do
    e, N = engine(), 1000
    p = voxels(N)
    a = predict(e, p)
    @test keys(a) == (:signal, :∂T₁, :∂T₂)
    @test all(A -> A isa CuMatrix{ComplexF32} && size(A) == (600, N), a)
    synchronize()
    ref = map(Array, a)
    @test all(A -> all(isfinite, A) && maximum(abs, A) > 0, ref)
    out = outputs(e, N)
    @test predict!(out, e, p) === out                           # returns the arrays it was given
    synchronize()
    @test map(Array, out) == ref                                # bitwise
    for _ in 1:3                                                # repeated calls: bitwise
        predict!(out, e, p)
        synchronize()
        @test map(Array, out) == ref
    end
    o1, o2 = outputs(e, N), outputs(e, N)                       # alternating output sets: bitwise
    for i in 1:4
        predict!(isodd(i) ? o1 : o2, e, p)
    end
    synchronize()
    @test map(Array, o1) == map(Array, o2) == ref
end

tier(:T2, "predict: signal only") do
    e, N = engine(), 777
    p = voxels(N; seed = 3)
    full = predict(e, p)
    s = predict(e, p; derivatives = false)
    @test keys(s) == (:signal,)
    @test keys(outputs(e, N; derivatives = false)) == (:signal,)
    synchronize()
    @test Array(s.signal) == Array(full.signal)                 # bitwise
    # A smaller batch after a larger one reuses the engine's scratch; the
    # first 10 voxels are the same ones, and results do not depend on the
    # batch they are computed in (dev/NOTES.md, S6 ③b).
    small = predict(e, voxels(10; seed = 3); derivatives = false)
    synchronize()
    @test Array(small.signal) == Array(full.signal)[:, 1:10]
end

tier(:T2, "predict: chunks into column views") do
    e, N = engine(), 14400
    p = voxels(N; seed = 5)
    whole = predict(e, p)
    chunked = outputs(e, N)
    for r in (1:4000, 4001:10000, 10001:14400)                  # boundaries at multiples of 4 (S6)
        predict!(map(A -> view(A, :, r), chunked), e, map(v -> view(v, r), p))
    end
    synchronize()
    @test map(Array, chunked) == map(Array, whole)              # bitwise (S6 ③b)
end

tier(:T2, "predict: bad outputs are refused before TensorRT runs") do
    e, N = engine(), 100
    p = voxels(N)
    good() = (signal = sentinel(600, N), ∂T₁ = sentinel(600, N), ∂T₂ = sentinel(600, N))
    cases = [  # out => words the ArgumentError must contain
        (o = good(); (; o..., ∂T₁ = sentinel(600, N + 1))) => "of size (600, 101)",
        (o = good(); (; o..., signal = CuMatrix{ComplexF64}(undef, 600, N))) => "CuMatrix{ComplexF32}",
        (o = good(); (; o..., ∂T₂ = view(sentinel(700, N), 1:600, :))) => "contiguous",
        (o = good(); (; o..., ∂T₂ = o.signal)) => "overlap",
        (o = good(); (; o..., signal = zeros(ComplexF32, 600, N))) => "CPU",
        (o = good(); (; o..., ∂T₁ = view(sentinel(600, N + 1), :, 2:N+1))) => "256",
        (o = good(); (; o.signal, o.∂T₁)) => "(; signal) or (; signal, ∂T₁, ∂T₂)",
    ]
    for (out, words) in cases
        err = @test_throws ArgumentError predict!(out, e, p)
        @test occursin(words, sprint(showerror, err.value))
        for A in values(out)                                    # nothing was written anywhere
            A isa CuMatrix{ComplexF32} && @test untouched(A)
        end
    end
end

tier(:T2, "predict: bad inputs and batch sizes are refused") do
    e = engine()
    out = (; signal = sentinel(600, 10))
    err = @test_throws ArgumentError predict!(out, e, voxels_cpu(10))
    @test occursin("CPU memory", sprint(showerror, err.value))
    p = voxels(10)
    err = @test_throws ArgumentError predict!(out, e, (; p.T₁, p.T₂, B₁ = CuArray(zeros(Float32, 9))))
    @test occursin("lengths", sprint(showerror, err.value))
    err = @test_throws ArgumentError predict!(out, e, (; T₁ = view(p.T₁, 10:-1:1), p.T₂, p.B₁))
    @test occursin("stride", sprint(showerror, err.value))
    @test untouched(out.signal)

    small = engine(; batch = (1, 64, 1000))
    err = @test_throws ArgumentError predict!(outputs(small, 1001), small, voxels(1001))
    msg = sprint(showerror, err.value)
    @test occursin("1001 voxels", msg) && occursin("1 to 1000", msg)
    @test occursin("batch = ", msg)                            # how to get an engine that fits
end

tier(:T2, "predict: results are in stream order, without synchronizing") do
    e, N = engine(), 14400
    p = voxels(N; seed = 7)
    ref = map(Array, predict(e, p))
    out = outputs(e, N)
    for _ in 1:3
        predict!(out, e, p)
        doubled = 2 .* out.signal                              # queued right behind, same stream
        synchronize()
        @test Array(doubled) == 2 .* ref.signal
    end
end

tier(:T2, "predict: no allocations after warm-up") do
    e, N = engine(), 14400
    p = voxels(N)
    sa = StructArray((; p.T₁, p.T₂, p.B₁, ρˣ = CuArray(ones(Float32, N)), ρʸ = CuArray(zeros(Float32, N))))
    for out in (outputs(e, N), outputs(e, N; derivatives = false)), q in (p, sa)
        predict!(out, e, q); predict!(out, e, q)
        synchronize()
        @test (@allocated predict!(out, e, q)) == 0               # host
        @test (CUDACore.@allocated predict!(out, e, q)) == 0      # GPU
    end
    synchronize()
end

tier(:T2, "predict: MRSTAT's StructArray, as is") do
    e, N = engine(), 500
    p = voxels(N; seed = 11)
    sa = StructArray((; p.T₁, p.T₂, p.B₁, ρˣ = CuArray(ones(Float32, N)), ρʸ = CuArray(zeros(Float32, N))))
    a, b = predict(e, p), predict(e, sa)
    synchronize()
    @test map(Array, a) == map(Array, b)
end

tier(:T2, "predict: two tasks share one engine") do
    e, N = engine(), 20000
    pA, pB = voxels(N; seed = 1), voxels(N; seed = 2)
    refA, refB = map(Array, predict(e, pA)), map(Array, predict(e, pB))
    synchronize()
    results = Dict{Symbol,Bool}()
    @sync for (name, p, ref) in ((:A, pA, refA), (:B, pB, refB))
        @async begin                                           # each task has its own stream
            out = outputs(e, N)
            ok = true
            for _ in 1:5
                predict!(out, e, p)
                yield()                                        # let the other task queue its run
                synchronize()
                ok &= map(Array, out) == ref
            end
            results[name] = ok
        end
    end
    @test results[:A] && results[:B]
end

tier(:T2, "predict: a small engine for small problems") do
    small, e = engine(; batch = (1, 64, 1000)), engine()
    @test batchprofile(small) == (min = 1, opt = 64, max = 1000)
    p = voxels(64; seed = 13)
    a, b = predict(small, p), predict(e, p)
    synchronize()
    @test all(relative(a[k], b[k]) ≤ 5e-6 for k in keys(a))    # another build of the same ONNX (D18)
end

tier(:T2, "predict: close") do
    e = Engine(; batch = (1, 64, 1000))
    p = voxels(8)
    predict(e, p)
    close(e)
    @test !isopen(e)
    close(e)                                                    # idempotent
    @test_throws InvalidStateException predict(e, p)
    foreach(close, values(ENGINES))
    empty!(ENGINES)
end
