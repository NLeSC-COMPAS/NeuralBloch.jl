# L2: the Engine — the model's TensorRT engine on this GPU — and predict!.

"""
    Engine(model = Model(); batch = (1, 14400, 40000), tf32 = false)

The model's TensorRT engine for the calling task's GPU, ready for
[`predict!`](@ref).

The first time on a machine (and after a TensorRT, GPU or setting change)
this compiles NeuralBloch's C++ against the system TensorRT and builds the
engine in a subprocess, a few seconds, saying so; both go into the cache
and later constructions load them.

- `batch = (min, opt, max)`: the batch sizes the engine accepts, and the one
  it is tuned for. The execution context holds ≈ 11 KiB of GPU scratch per
  voxel of `max` (440 MiB at the default); batches outside the range are an
  error, never padded.
- `tf32`: allow TF32 inner products (Ampere and newer; accuracy first, so off).

Throws a `PrerequisiteError` when the system lacks something (see
[`NeuralBloch.versioninfo`](@ref)), a `BuildError` when building fails, and
a [`BundleError`](@ref) when the engine does not match the model card.
`close` frees the engine's GPU memory; otherwise the garbage collector does.
"""
mutable struct Engine
    const model::Model
    const handle::TensorRT.Handle
    const io::@NamedTuple{x::Int, signal::Int, ∂T₁::Int, ∂T₂::Int}   # tensor indices
    const batch::@NamedTuple{min::Int, opt::Int, max::Int}
    const gpu::String
    const lock::ReentrantLock
    # Concrete types (CuMatrix{T} leaves the memory type open, and an abstract
    # field makes every call on it dynamic, which allocates).
    const input::CuArray{Float32,2,DeviceMemory}     # (3, max): [T₁; T₂; B₁] per voxel, the engine's x
    sink::CuArray{ComplexF32,2,DeviceMemory}         # (NTR, n): the derivatives when only the signal is wanted
end

function Engine(model::Model = Model(); batch = (1, 14400, 40000), tf32::Bool = false)
    settings = TensorRT.BuildSettings(; batch, tf32)
    sys = TensorRT.detect()
    handle = TensorRT.Handle(TensorRT.build(model.onnx; settings, sys); native = TensorRT.compile(sys))
    io = try
        bindings(TensorRT.tensors(handle), model.ntr)
    catch
        close(handle)
        rethrow()
    end
    lo, opt, hi = (shape[1] for shape in TensorRT.tensors(handle)[io.x+1].profile)
    return Engine(model, handle, io, (min = lo, opt, max = hi), sys.gpu.name, ReentrantLock(),
                  CuMatrix{Float32}(undef, 3, hi), CuMatrix{ComplexF32}(undef, model.ntr, 0))
end

Base.show(io::IO, e::Engine) = print(io, "Engine(NTR ", e.model.ntr, ", batch ", e.batch.min, "–", e.batch.max,
                                     " (opt ", e.batch.opt, "), ", e.gpu, isopen(e) ? "" : ", closed", ")")

"""
    close(engine::Engine)

Free the engine's GPU memory, TensorRT's and its own buffers, once the runs
already queued have finished. Safe to call more than once and from any
task; `predict!` calls after it, or waiting for it, throw an
`InvalidStateException`.
"""
function Base.close(e::Engine)
    @lock e.lock begin
        if isopen(e.handle)
            close(e.handle)
            CUDACore.unsafe_free!(e.input)
            CUDACore.unsafe_free!(e.sink)
        end
    end
    return nothing
end
Base.isopen(e::Engine) = isopen(e.handle)

# --- metadata -----------------------------------------------------------------------

"""
    ntr(model_or_engine) -> Int

Readouts per voxel: the rows of every output.
"""
ntr(m::Model) = m.ntr
ntr(e::Engine) = ntr(e.model)

"""
    domain(model_or_engine) -> (; T₁, T₂, B₁)

The training domain, `(low, high)` per property: T₁ and T₂ in seconds, B₁
relative. The network was also only trained on T₂ < T₁; outside, it
extrapolates. See [`checkdomain`](@ref).
"""
domain(m::Model) = NamedTuple{FEATURES}(map(name -> m.domain[Symbol(cardname(name))], FEATURES))
domain(e::Engine) = domain(e.model)

"""
    batchprofile(engine) -> (; min, opt, max)

The batch sizes `engine` accepts (`min` to `max` voxels per call), and the
one it is tuned for.
"""
batchprofile(e::Engine) = e.batch

"""
    sequence(model_or_engine) -> NamedTuple

The FISP3D sequence the model's training data was simulated with: TR, TE,
TI, TW (seconds), max_state, repetitions, inversion_prepulse, wait_spoiling,
undersampling_factor, delta_k_spoil (rad/m), and the RF train as rf_angle
and rf_phase (degrees, one per readout).
"""
sequence(m::Model) = m.sequence
sequence(e::Engine) = sequence(e.model)

# --- predicting ---------------------------------------------------------------------

"""
    outputs(engine, N; derivatives = true)

Fresh output arrays for a batch of `N` voxels: `(; signal, ∂T₁, ∂T₂)`, or
`(; signal)` without derivatives, each an `(NTR, N)` `CuMatrix{ComplexF32}`.
"""
function outputs(engine::Engine, N::Integer; derivatives::Bool = true)
    new() = CuMatrix{ComplexF32}(undef, engine.model.ntr, N)
    return derivatives ? (signal = new(), ∂T₁ = new(), ∂T₂ = new()) : (; signal = new())
end

"""
    predict!(out, engine, p) -> out

Predict, for every voxel of `p`, the signal and (if `out` has them) its
derivatives, writing them into `out` in place.

- `p`: anything with properties `T₁`, `T₂`, `B₁` holding GPU vectors of
  Float32 of equal length N, in physical units (T₁ and T₂ in seconds, B₁
  relative), e.g. MRSTAT's StructArray as it is. Vectors may be strided
  views (positive stride).
- `out`: `(; signal)` or `(; signal, ∂T₁, ∂T₂)`, each an `(NTR, N)`
  `CuMatrix{ComplexF32}` indexed `[readout, voxel]`: the signal at unit
  proton density, ∂T₁ and ∂T₂ in physical units (per second), not scaled.
  The caller owns them: any address with TensorRT's 256-byte alignment
  (whole arrays have it; see the error for column views), no two
  overlapping. Only the signal requested: the derivatives go to scratch
  memory the engine owns.
- `N` must be in [`batchprofile(engine)`](@ref), or 0 (then nothing happens).

Asynchronous: everything is queued on the calling task's stream and ready
in stream order, without waiting for the GPU (so it does not check the
training domain; see [`checkdomain`](@ref)). One exception: when the
previous call came from another task, this one first waits for that task's
stream, since the engine's buffers change hands. Keep `out` and `p` alive
until the results are used. Every contract violation is an `ArgumentError`,
thrown before anything is queued. After warm-up it allocates nothing, on
the host or the GPU.
"""
function predict!(out, engine::Engine, p)
    N = checkcall(engine, p)
    checkoutputs(out, engine.model.ntr, N)
    N == 0 && return out
    h, io = engine.handle, engine.io
    @lock engine.lock begin
        stream = CUDACore.stream()
        TensorRT.claim!(h, stream)
        stage!(engine.input, p, N, stream)
        TensorRT.set_shape!(h, io.x, (N, 3))
        TensorRT.set_address!(h, io.x, devptr(engine.input))
        TensorRT.set_address!(h, io.signal, devptr(out.signal))
        ∂T₁, ∂T₂ = derivatives(engine, out, N)
        TensorRT.set_address!(h, io.∂T₁, devptr(∂T₁))
        TensorRT.set_address!(h, io.∂T₂, devptr(∂T₂))
        TensorRT.run!(h, stream)
    end
    return out
end

"""
    predict(engine, p; derivatives = true)

[`predict!`](@ref) into fresh [`outputs`](@ref).
"""
predict(engine::Engine, p; derivatives::Bool = true) =
    predict!(outputs(engine, checkcall(engine, p); derivatives), engine, p)

# The batch size of a call on `engine` with inputs `p`, once both are known to
# be fine: before outputs are allocated for it, which a batch too large for
# the engine could fail at.
function checkcall(engine::Engine, p)
    isopen(engine) || throw(InvalidStateException("the engine is closed", :closed))
    N = checkinputs(p)
    problem = batch_problem(engine.batch, engine.model.ntr, N)
    problem === nothing || throw(ArgumentError(problem))
    return N
end

# Taking the pointer hands the array to the task's stream (CUDA.jl orders
# its free after our work); TensorRT then gets the bare address.
devptr(A::CuArray) = reinterpret(Ptr{Cvoid}, pointer(A))

# Where the derivatives go: the caller's arrays or, when only the signal is
# wanted, the engine's sink, one buffer for both (dev/NOTES.md, S6).
derivatives(engine::Engine, out::NamedTuple{(:signal,)}, N) = (s = sink!(engine, N); (s, s))
derivatives(engine::Engine, out::NamedTuple, N) = (out.∂T₁, out.∂T₂)

# The sink grows to the largest batch seen, never shrinks: after warm-up,
# no allocation. Its first N columns are what TensorRT writes.
function sink!(engine::Engine, N)
    size(engine.sink, 2) < N && (engine.sink = CuMatrix{ComplexF32}(undef, engine.model.ntr, N))
    return engine.sink
end

# Copy p's vectors into the rows of the input buffer, on `stream`: three
# strided device-to-device copies, which allocate nothing (dev/NOTES.md,
# S10). Row k of the (3, max) buffer is every third Float32.
function stage!(input::CuMatrix{Float32}, p, N, stream)
    stage_row!(input, 1, p.T₁, N, stream)
    stage_row!(input, 2, p.T₂, N, stream)
    stage_row!(input, 3, p.B₁, N, stream)
end

stage_row!(input, row, v, N, stream) =
    CUDACore.unsafe_copy2d!(pointer(input, row), CUDACore.DeviceMemory, pointer(v), CUDACore.DeviceMemory,
                            1, N; dstPitch = 3 * sizeof(Float32), srcPitch = stride(v, 1) * sizeof(Float32),
                            async = true, stream)

# --- the training domain ------------------------------------------------------------

"""
    checkdomain(model_or_engine, p) -> p

Throw an `ArgumentError` unless every voxel of `p` lies in the training
domain ([`domain`](@ref)): each of `p.T₁`, `p.T₂`, `p.B₁` that `p` has
within its range (NaN is not), and T₂ < T₁ where `p` has both. Outside,
the network extrapolates.

`p` may hold GPU or CPU vectors. For GPU vectors this reads reductions back
to the host, so it synchronizes: that is why [`predict!`](@ref) does not
call it.
"""
function checkdomain(model::Model, p)
    present = filter(name -> hasproperty(p, name), FEATURES)
    isempty(present) && throw(ArgumentError("p has none of the properties T₁, T₂, B₁ that checkdomain checks"))
    for name in present
        v = getproperty(p, name)
        lo, hi = model.domain[Symbol(cardname(name))]
        vmin, vmax = extrema(v)
        within(eltype(v), lo, hi, vmin, vmax) && continue
        throw(ArgumentError("p.$name ranges over [$vmin, $vmax], outside the training domain [$lo, $hi]; " *
                            "the network extrapolates there"))
    end
    if model.constraint == "T2 < T1" && hasproperty(p, :T₁) && hasproperty(p, :T₂)
        n = count(p.T₂ .>= p.T₁)
        n == 0 || throw(ArgumentError(
            "$n voxels have T₂ ≥ T₁, outside the training domain (T₂ < T₁ only); the network extrapolates there"))
    end
    return p
end

checkdomain(engine::Engine, p) = checkdomain(engine.model, p)

# [vmin, vmax] ⊆ [lo, hi], with the bounds in the inputs' precision, rounded
# outward: a bound rounded to Float32 is inside (0.01f0 < 0.01, yet it is the
# bound as MRSTAT stores it).
within(::Type{T}, lo, hi, vmin, vmax) where {T<:AbstractFloat} = T(lo, RoundDown) ≤ vmin && vmax ≤ T(hi, RoundUp)
within(::Type, lo, hi, vmin, vmax) = lo ≤ vmin && vmax ≤ hi

# --- reporting ----------------------------------------------------------------------

"""
    NeuralBloch.versioninfo([io])

What NeuralBloch uses on this machine: the bundled model, the cache, and the
system prerequisites (TensorRT's libraries and headers, the CUDA headers,
g++, the GPU), with the apt/dnf commands that install what is missing.
"""
function versioninfo(io::IO = stdout)
    println(io, "NeuralBloch ", pkgversion(@__MODULE__))
    model = try
        m = Model()
        "$(m.onnx) (sha256 $(m.sha256[1:12])…, NTR $(m.ntr), $(m.provenance["upstream"]))"
    catch err
        err isa BundleError || rethrow()
        sprint(showerror, err)
    end
    println(io, "  model: ", model)
    println(io, "  cache: ", TensorRT.cache_root(), "\n")
    TensorRT.versioninfo(io)
end
