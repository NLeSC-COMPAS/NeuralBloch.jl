# engine.jl + engine.cpp: opening engines and running them.

using NeuralBloch.TensorRT: Handle, TensorRTError, build, BuildSettings, tensors, tensorindex,
                            set_shape!, set_address!, run!, claim!, scratch_bytes
using CUDACore: CUDACore, CuArray, CuMatrix, CuStream, synchronize

# (3, N) Float32 inputs [T₁; T₂; B₁] inside the training domain, reproducibly.
function inputs(N; seed = 1)
    X = zeros(Float32, 3, N)
    for j in 1:N
        u = mod.(seed .* (0.618034, 0.414214, 0.732051) .* j, 1)
        T₁ = exp(log(0.1) + u[1] * log(50))
        X[:, j] .= (T₁, min(exp(log(0.01) + u[2] * log(200)), 0.9T₁), 0.8 + 0.4u[3])
    end
    return CuArray(X)
end

devptr(A) = reinterpret(Ptr{Cvoid}, pointer(A))

# Bind x and the three outputs of `h` for a batch of size(X, 2); return the outputs.
function bind!(h, X, outs = ntuple(_ -> CuMatrix{ComplexF32}(undef, 600, size(X, 2)), 3))
    set_shape!(h, tensorindex(h, "x"), (size(X, 2), 3))
    set_address!(h, tensorindex(h, "x"), devptr(X))
    for (name, A) in zip(("signal", "dT1", "dT2"), outs)
        set_address!(h, tensorindex(h, name), devptr(A))
    end
    return outs
end

tier(:T2, "engine: a file TensorRT cannot read") do
    mktempdir() do dir
        missing = joinpath(dir, "missing.plan")
        err = @test_throws TensorRTError Handle(missing)
        @test occursin("cannot read $missing", sprint(showerror, err.value))
        garbage = joinpath(dir, "garbage.plan")
        write(garbage, rand(UInt8, 4096))
        err = @test_throws TensorRTError Handle(garbage)
        @test occursin("cannot deserialize", sprint(showerror, err.value))
    end
end

tier(:T2, "engine: I/O description") do
    h = Handle(build(ONNX))
    ts = tensors(h)
    @test sort([t.name for t in ts]) == ["dT1", "dT2", "signal", "x"]
    x = ts[tensorindex(h, "x") + 1]
    @test x.input && x.eltype === Float32
    @test x.shape == [-1, 3]                                # TensorRT order: (batch, features)
    @test x.profile == ([1, 3], [14400, 3], [40000, 3])
    for name in ("signal", "dT1", "dT2")
        t = ts[tensorindex(h, name) + 1]
        @test !t.input && t.eltype === Float32
        @test t.shape == [-1, 600, 2]                       # (batch, NTR, re/im)
        @test t.profile == (Int[], Int[], Int[])
    end
    @test tensorindex(h, "nonexistent") === nothing
    @test 400 * 2^20 < scratch_bytes(h) < 500 * 2^20      # ≈ 439 MiB (dev/NOTES.md, S4)
    close(h)
end

tier(:T2, "engine: batches outside the profile") do
    h = Handle(build(ONNX))
    x = tensorindex(h, "x")
    for bad in ((0, 3), (40001, 3), (100, 4), (100,))
        err = @test_throws ArgumentError set_shape!(h, x, bad)
        msg = sprint(showerror, err.value)
        @test occursin("outside the engine's optimization profile", msg)
        @test occursin("(1, 3)", msg) && occursin("(40000, 3)", msg)   # the range it accepts
    end
    err = @test_throws ArgumentError set_shape!(h, tensorindex(h, "signal"), (10, 600, 2))
    @test occursin("output", sprint(showerror, err.value))
    set_shape!(h, x, (1, 3))                               # both ends are inside
    set_shape!(h, x, (40000, 3))
    close(h)
end

tier(:T2, "engine: TensorRT's refusals reach the user") do
    h = Handle(build(ONNX))
    X = inputs(10)
    err = @test_throws TensorRTError set_address!(h, length(tensors(h)), devptr(X))
    @test occursin("rejected the address of an invalid tensor index", sprint(showerror, err.value))
    set_shape!(h, tensorindex(h, "x"), (10, 3))
    err = @test_throws TensorRTError run!(h)                   # no address set
    msg = sprint(showerror, err.value)
    @test occursin("could not enqueue the engine", msg)
    @test occursin("Address is not set for input tensor x", msg)  # TensorRT's own words
    close(h)
end

tier(:T2, "engine: runs, closes, coexists") do
    path = build(ONNX)
    h1, h2 = Handle(path), Handle(path)                    # two coexist
    small = Handle(build(ONNX; settings = BuildSettings(; batch = (1, 64, 1000))))
    X = inputs(1000)
    o1, o2, o3 = bind!(h1, X), bind!(h2, X), bind!(small, X)
    run!(h1); run!(h2); run!(small)
    synchronize()
    for k in 1:3
        a = Array(o1[k])
        @test all(isfinite, a) && maximum(abs, a) > 0
        @test Array(o2[k]) == a                            # same engine: bitwise
        @test maximum(abs, Array(o3[k]) .- a) ≤ 5e-6 * maximum(abs, a)   # another build (S4, D18)
    end

    # Closing is idempotent; afterwards every call fails cleanly.
    close(h1)
    @test !isopen(h1) && isopen(h2)
    close(h1)
    @test_throws InvalidStateException run!(h1)
    @test_throws InvalidStateException set_shape!(h1, 0, (10, 3))
    @test_throws InvalidStateException set_address!(h1, 0, devptr(X))
    @test_throws InvalidStateException claim!(h1)
    run!(h2)                                               # the other one is unaffected
    synchronize()
    @test Array(o2[1]) == Array(o1[1])
    close(h2); close(small)
end

tier(:T2, "engine: runs on different streams are ordered") do
    h = Handle(build(ONNX))
    N = 40000                                              # long runs, so they would overlap
    XA, XB = inputs(N; seed = 1), inputs(N; seed = 2)
    outs = bind!(h, XA); run!(h); synchronize()
    refA = Array.(outs)
    outs = bind!(h, XB); run!(h); synchronize()
    refB = Array.(outs)
    for _ in 1:5
        sA, sB = CuStream(), CuStream()
        A = bind!(h, XA); claim!(h, sA); run!(h, sA)
        B = bind!(h, XB); claim!(h, sB); run!(h, sB)       # shares the context's scratch with A
        synchronize(sA); synchronize(sB)
        @test Array.(A) == refA && Array.(B) == refB
    end
    close(h)
end

tier(:T2, "engine: running allocates nothing on the host") do
    h = Handle(build(ONNX))
    X = inputs(14400)
    outs = bind!(h, X)
    x, s = tensorindex(h, "x"), CUDACore.stream()
    p = devptr(outs[1])
    signal = tensorindex(h, "signal")
    step() = (claim!(h, s); set_shape!(h, x, (14400, 3)); set_address!(h, signal, p); run!(h, s))
    step(); step()
    synchronize()
    @test (@allocated step()) == 0
    synchronize()
    close(h)
end
