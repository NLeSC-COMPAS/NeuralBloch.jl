# predict.jl: Engine, outputs, predict!, predict, checkdomain and the
# metadata queries. The T2 testsets follow the predict! checklist of the plan,
# for each target: the signal and its derivatives, the signal alone, and the
# basis coefficients of the signal. Each target has its own engine.

using NeuralBloch: Engine, Model, outputs, predict!, predict, checkdomain, ntr, rank, domain, batchprofile,
                   sequence, TARGETS
using CUDACore: CUDACore, CuArray, CuMatrix, CuVector, synchronize
using LinearAlgebra: norm
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

sentinel(::Type{T}, dims::Integer...) where {T} = fill!(CuMatrix{T}(undef, dims...), T(NaN))
sentinel(dims::Integer...) = sentinel(ComplexF32, dims...)
untouched(A) = (synchronize(); all(isnan, Array(A)))
relative(a, b) = maximum(abs, Array(a) .- Array(b)) / maximum(abs, Array(b))

# The output names, the element type and the rows of each target.
const SETS = (signal_and_derivatives = ((:signal, :∂T₁, :∂T₂), ComplexF32, 600),
              signal = ((:signal,), ComplexF32, 600),
              coefficients = ((:coefficients,), Float32, 16))

tier(:T0, "predict: model metadata and checkdomain") do
    m = Model()
    @test ntr(m) == 600
    @test rank(m) == 16
    @test domain(m) == (T₁ = (0.1, 6.0), T₂ = (0.01, 2.0), B₁ = (0.5, 1.5))
    s = sequence(m)
    @test s.kind == "FISP3D" && s.TR == 0.00558 && length(s.rf_angle) == 600

    p = voxels_cpu(100)
    @test checkdomain(m, p) === p                               # works on CPU vectors too
    @test checkdomain(m, (; B₁ = p.B₁)) isa NamedTuple          # any subset of the properties
    for (name, bad, words) in ((:T₁, 6.5f0, "p.T₁ ranges over"), (:T₁, 0.05f0, "[0.1, 6.0]"),
                               (:T₂, 2.5f0, "p.T₂"), (:B₁, 1.55f0, "[0.5, 1.5]"), (:B₁, 0.45f0, "[0.5, 1.5]"),
                               (:B₁, NaN32, "p.B₁"))
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

    # The bounds themselves are inside, also once rounded to Float32 (which
    # puts 0.01f0 below 0.01): MRSTAT clamps its parameters to domain().
    edges = (T₁ = Float32[0.1, 6.0], T₂ = Float32[0.01, 2.0], B₁ = Float32[0.5, 1.5])
    @test checkdomain(m, edges) === edges
    @test checkdomain(m, (T₁ = [0.1, 6.0], T₂ = [0.01, 2.0], B₁ = [0.5, 1.5])) isa NamedTuple
    below = (; edges..., T₂ = [prevfloat(Float32(0.01, RoundDown)), 2.0f0])
    err = @test_throws ArgumentError checkdomain(m, below)
    @test occursin("p.T₂", sprint(showerror, err.value))
end

tier(:T2, "predict: engine metadata") do
    e = engine()
    @test ntr(e) == 600
    @test rank(e) == 16
    @test batchprofile(e) == (min = 1, opt = 14400, max = 40000)
    @test domain(e) == domain(Model())
    @test sequence(e) == sequence(Model())
    @test isopen(e)
    @test keys(e.targets) == TARGETS
    @test occursin("NTR 600, rank 16", sprint(show, e))
    p = voxels(10)
    @test checkdomain(e, p) === p                               # and on the GPU
end

tier(:T2, "predict: predict and predict! agree, repeatedly, for every target") do
    e, N = engine(), 1000
    p = voxels(N)
    for (target, (names, E, rows)) in pairs(SETS)
        a = predict(e, p; target)
        @test keys(a) == names
        @test all(A -> A isa CuMatrix{E} && size(A) == (rows, N), a)
        @test keys(outputs(e, N; target)) == names
        synchronize()
        ref = map(Array, a)
        @test all(A -> all(isfinite, A) && maximum(abs, A) > 0, ref)
        out = outputs(e, N; target)
        @test predict!(out, e, p) === out                       # returns the arrays it was given
        synchronize()
        @test map(Array, out) == ref                            # bitwise
        for _ in 1:3                                            # repeated calls: bitwise
            predict!(out, e, p)
            synchronize()
            @test map(Array, out) == ref
        end
        o1, o2 = outputs(e, N; target), outputs(e, N; target)   # alternating output sets: bitwise
        for i in 1:4
            predict!(isodd(i) ? o1 : o2, e, p)
        end
        synchronize()
        @test map(Array, o1) == map(Array, o2) == ref
    end
    @test predict(e, p) isa NamedTuple{(:signal, :∂T₁, :∂T₂)}  # the default target
    o = outputs(e, N)
    reordered = (∂T₂ = o.∂T₂, signal = o.signal, ∂T₁ = o.∂T₁)   # any order of the three
    @test predict!(reordered, e, p) === reordered
    synchronize()
    @test map(Array, o) == map(Array, predict(e, p))
end

tier(:T2, "predict: targets interleaved in one task") do
    e, N = engine(), 2000
    p = voxels(N; seed = 4)
    refs = map(target -> map(Array, predict(e, p; target)), TARGETS)
    outs = map(target -> outputs(e, N; target), TARGETS)
    for _ in 1:3, out in outs
        predict!(out, e, p)
    end
    synchronize()
    @test map(o -> map(Array, o), outs) == refs
end

tier(:T2, "predict: the signal alone, from its own engine") do
    e, N = engine(), 777
    p = voxels(N; seed = 3)
    full = predict(e, p)
    s = predict(e, p; target = :signal)
    synchronize()
    # A different engine, thus the bound between builds applies (D18). On the
    # RTX 2080 Ti with TensorRT 11.2, the kernels are the same and the results
    # are bitwise equal.
    @test relative(s.signal, full.signal) ≤ 5e-6
    # A smaller batch after a larger batch. The first 10 voxels are the same,
    # and the results do not change with the batch (dev/NOTES.md, S6 ③b).
    small = predict(e, voxels(10; seed = 3); target = :signal)
    synchronize()
    @test Array(small.signal) == Array(s.signal)[:, 1:10]
end

tier(:T2, "predict: the coefficients are the signal's, in a rank-16 basis") do
    # The bundle does not include the basis, but the signal must be one linear
    # map of the coefficients. A least-squares fit of that map on many voxels
    # gives the signal to float precision only if the two engines are the same
    # network.
    e, N = engine(), 2000
    p = voxels(N; seed = 9)
    C = Float64.(Array(predict(e, p; target = :coefficients).coefficients))
    S = ComplexF64.(Array(predict(e, p).signal))
    @test size(C) == (16, N)
    U = S / C                                                   # (600, 16), least squares
    @test norm(S - U * C) / norm(S) ≤ 1e-5
    # The coefficients of other voxels: the same map, from the fit on the first set
    q = voxels(500; seed = 10)
    Cq = Float64.(Array(predict(e, q; target = :coefficients).coefficients))
    Sq = ComplexF64.(Array(predict(e, q; target = :signal).signal))
    @test norm(Sq - U * Cq) / norm(Sq) ≤ 1e-5
end

tier(:T2, "predict: chunks into column views") do
    e, N = engine(), 14400
    p = voxels(N; seed = 5)
    for target in TARGETS
        whole = predict(e, p; target)
        chunked = outputs(e, N; target)
        for r in (1:4000, 4001:10000, 10001:14400)              # boundaries at multiples of 4 (S6)
            predict!(map(A -> view(A, :, r), chunked), e, map(v -> view(v, r), p))
        end
        synchronize()
        @test map(Array, chunked) == map(Array, whole)          # bitwise (S6 ③b)
    end
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
        (o = good(); (; o.signal, o.∂T₁)) => "(; signal, ∂T₁, ∂T₂) in any order, (; signal) or (; coefficients)",
        (o = good(); (; o.signal, coefficients = sentinel(Float32, 16, N))) => "(; coefficients)",
        (; coefficients = sentinel(16, N)) => "CuMatrix{Float32}",            # complex for real
        (; coefficients = sentinel(Float32, 600, N)) => "a batch of 100 voxels needs (16, 100)",
        (; coefficients = view(sentinel(Float32, 16, N + 1), :, 2:N+1)) => "multiple of 4",
        (; signal = sentinel(Float32, 600, N)) => "CuMatrix{ComplexF32}",         # real for complex
    ]
    for (out, words) in cases
        err = @test_throws ArgumentError predict!(out, e, p)
        @test occursin(words, sprint(showerror, err.value))
        for A in values(out)                                    # nothing was written to the sentinels
            (A isa CuMatrix{ComplexF32} || A isa CuMatrix{Float32}) && @test untouched(A)
        end
    end
    @test_throws ArgumentError outputs(e, N; target = :derivatives)
    err = @test_throws ArgumentError predict(e, p; target = :derivatives)
    @test occursin("target must be :signal_and_derivatives, :signal or :coefficients", sprint(showerror, err.value))
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
    for target in TARGETS
        err = @test_throws ArgumentError predict!(outputs(small, 1001; target), small, voxels(1001))
        msg = sprint(showerror, err.value)
        @test occursin("1001 voxels", msg) && occursin("1 to 1000", msg)
        @test occursin("batch = ", msg)                        # how to get an engine that fits
    end

    # predict refuses before allocating outputs a batch that size would need.
    large = voxels(2000)
    for target in TARGETS
        @test (CUDACore.@allocated try predict(small, large; target) catch end) == 0
        @test_throws ArgumentError predict(small, large; target)
    end

    # An empty batch (a mask with no voxels) is nothing to do.
    empty = voxels(0)
    for (target, (names, E, rows)) in pairs(SETS)
        out0 = outputs(small, 0; target)
        @test predict!(out0, small, empty) === out0
        @test size(first(predict(small, empty; target))) == (rows, 0)
    end
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
    o = outputs(e, N)
    reordered = (∂T₂ = o.∂T₂, signal = o.signal, ∂T₁ = o.∂T₁)
    for out in (o, reordered, outputs(e, N; target = :signal), outputs(e, N; target = :coefficients)), q in (p, sa)
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
    # Each pair gives the target of each task. With the same target, the two
    # tasks share the buffers of one engine. With different targets, they
    # share only the Engine object.
    for (targetA, targetB) in ((:signal_and_derivatives, :signal_and_derivatives), (:signal_and_derivatives, :signal),
                               (:coefficients, :signal))
        refA, refB = predict(e, pA; target = targetA), predict(e, pB; target = targetB)
        synchronize()
        results = Dict{Symbol,Bool}()
        @sync for (name, p, target, ref) in ((:A, pA, targetA, refA), (:B, pB, targetB, refB))
            @async begin                                       # each task has its own stream
                out = outputs(e, N; target)
                ok = true
                for _ in 1:5
                    predict!(out, e, p)
                    yield()                                    # let the other task queue its run
                    synchronize()
                    ok &= all(map(==, out, ref))               # bitwise, compared on the GPU
                end
                results[name] = ok
            end
        end
        @test results[:A] && results[:B]
    end
end

tier(:T2, "predict: a small engine for small problems") do
    small, e = engine(; batch = (1, 64, 1000)), engine()
    @test batchprofile(small) == (min = 1, opt = 64, max = 1000)
    p = voxels(64; seed = 13)
    for target in TARGETS
        a, b = predict(small, p; target), predict(e, p; target)
        synchronize()
        @test all(relative(a[k], b[k]) ≤ 5e-6 for k in keys(a))    # another build of the same ONNX (D18)
    end
end

tier(:T2, "predict: close") do
    e = Engine(; batch = (1, 64, 1000))
    p = voxels(8)
    for target in TARGETS
        predict(e, p; target)
    end
    synchronize()
    own = sum(t -> sizeof(t.input), e.targets)
    used = CUDACore.used_memory()
    close(e)
    @test !isopen(e)
    @test all(t -> !isopen(t.handle), e.targets)                 # every engine
    @test used - CUDACore.used_memory() ≥ own                   # its buffers too, not only TensorRT's
    close(e)                                                    # idempotent
    for target in TARGETS
        @test_throws InvalidStateException predict(e, p; target)
    end
    @test_throws InvalidStateException predict!(outputs(engine(), 8; target = :coefficients), e, p)
end

tier(:T2, "predict: close races predict! and close") do
    # One thread predicts in a loop while two others close the engine: the
    # loop must end with InvalidStateException, and nothing may crash (a
    # use-after-free or a double free would take the process down).
    code = """
    using NeuralBloch, CUDACore
    N = 200
    p = (T₁ = CuArray(fill(1f0, N)), T₂ = CuArray(fill(0.1f0, N)), B₁ = CuArray(fill(1f0, N)))
    seen = Set{String}()
    for round in 1:10
        e = Engine(; batch = (1, 64, 1000))
        out = outputs(e, N; target = NeuralBloch.TARGETS[mod1(round, 3)])
        running = Threads.@spawn try
            while true
                predict!(out, e, p)
            end
        catch err
            err
        end
        sleep(0.05)
        foreach(wait, [Threads.@spawn close(e) for _ in 1:2])
        push!(seen, string(nameof(typeof(fetch(running)))))
    end
    print(join(sort(collect(seen)), ","))
    """
    cmd = `$(Base.julia_cmd()) -t 3 --startup-file=no --project=$(Base.active_project()) -e $code`
    @test readchomp(pipeline(cmd; stderr = devnull)) == "InvalidStateException"
    foreach(close, values(ENGINES))
    empty!(ENGINES)
end
