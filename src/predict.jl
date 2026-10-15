# L2: the Engine — the model's TensorRT engines on this GPU, one per target —
# and predict!.

"""
    Target

One target's engine: its TensorRT handle, the 0-based indices of its tensors
(`x` and the target's outputs, by public name), and its own input buffer.

Each target has its own buffer because `TensorRT.claim!` orders the runs of
one handle only: with one buffer for all, a run staged on one task's stream
could overwrite the input that another target's run, queued on another
stream, is still reading.
"""
struct Target{IO<:NamedTuple}
    handle::TensorRT.Handle
    io::IO
    # Concrete types (CuMatrix{T} leaves the memory type open, and an abstract
    # field makes every call on it dynamic, which allocates).
    input::CuArray{Float32,2,DeviceMemory}       # (3, max): [T₁; T₂; B₁] per voxel, the engine's x
end

const Targets = @NamedTuple{signal_and_derivatives::Target{@NamedTuple{x::Int, signal::Int, ∂T₁::Int, ∂T₂::Int}},
                            signal::Target{@NamedTuple{x::Int, signal::Int}},
                            coefficients::Target{@NamedTuple{x::Int, coefficients::Int}}}

"""
    Engine(model = Model(); batch = (1, 14400, 40000), tf32 = false)

The model's TensorRT engines for the calling task's GPU, one per target,
ready for [`predict!`](@ref): the signal and its derivatives, the signal
alone, and the signal's basis coefficients. Each computes only what its
target needs.

The first time on a machine (and after a TensorRT, GPU or setting change)
this compiles NeuralBloch's C++ against the system TensorRT and builds the
three engines in subprocesses, a few seconds each, saying so; all go into the
cache and later constructions load them.

- `batch = (min, opt, max)`: the batch sizes the engines accept, and the one
  they are tuned for. Each engine's execution context holds GPU scratch in
  proportion to `max`: ≈ 11 KiB per voxel for the signal and its derivatives
  (440 MiB at the default), less for the others (see `dev/NOTES.md`). Batches
  outside the range are an error, never padded.
- `tf32`: allow TF32 inner products (Ampere and newer; accuracy first, so off).

Throws a `PrerequisiteError` when the system lacks something (see
[`NeuralBloch.versioninfo`](@ref)), a `BuildError` when building fails, and
a [`BundleError`](@ref) when an engine does not match the model card.
`close` frees the engines' GPU memory; otherwise the garbage collector does.
"""
mutable struct Engine
    const model::Model
    const targets::Targets
    const batch::@NamedTuple{min::Int, opt::Int, max::Int}
    const gpu::String
    const lock::ReentrantLock
end

function Engine(model::Model = Model(); batch = (1, 14400, 40000), tf32::Bool = false)
    settings = TensorRT.BuildSettings(; batch, tf32)
    sys = TensorRT.detect()
    # Build every engine before opening any: an open execution context holds
    # GPU memory the builder subprocess may need. (build checks every
    # prerequisite first, the GPU included, and compiles what it needs.)
    paths = map(target -> TensorRT.build(model.onnx[target]; settings, sys), TARGETS)
    native = TensorRT.compile(sys)
    opened = TensorRT.Handle[]
    try
        handles = map(paths) do path
            h = TensorRT.Handle(path; native)
            push!(opened, h)
            h
        end
        ios = map((target, h) -> resolve_io(TensorRT.tensors(h), target; model.ntr, model.rank), TARGETS, handles)
        profiles = map((h, io) -> Tuple(shape[1] for shape in TensorRT.tensors(h)[io.x+1].profile), handles, ios)
        allequal(profiles) || throw(BundleError(
            "the engines of $(model.dir) accept different batch sizes: $(NamedTuple{TARGETS}(profiles))"))
        lo, opt, hi = first(profiles)
        targets = NamedTuple{TARGETS}(map((h, io) -> Target(h, io, CuMatrix{Float32}(undef, 3, hi)), handles, ios))
        return Engine(model, targets, (min = lo, opt, max = hi), sys.gpu.name, ReentrantLock())
    catch
        foreach(close, opened)
        rethrow()
    end
end

Base.show(io::IO, e::Engine) = print(io, "Engine(NTR ", e.model.ntr, ", rank ", e.model.rank, ", batch ", e.batch.min,
                                     "–", e.batch.max, " (opt ", e.batch.opt, "), ", e.gpu,
                                     isopen(e) ? "" : ", closed", ")")

"""
    close(engine::Engine)

Free the engines' GPU memory, TensorRT's and their own buffers, once the runs
already queued have finished. Safe to call more than once and from any
task; `predict!` calls after it, or waiting for it, throw an
`InvalidStateException`.
"""
function Base.close(e::Engine)
    @lock e.lock begin
        if isopen(e)
            for t in e.targets
                close(t.handle)
                CUDACore.unsafe_free!(t.input)
            end
        end
    end
    return nothing
end
Base.isopen(e::Engine) = isopen(e.targets.signal_and_derivatives.handle)   # all close together

# --- metadata -----------------------------------------------------------------------

"""
    ntr(model_or_engine) -> Int

Readouts per voxel: the rows of the signal and its derivatives.
"""
ntr(m::Model) = m.ntr
ntr(e::Engine) = ntr(e.model)

"""
    rank(model_or_engine) -> Int

Basis coefficients per voxel: the rows of the `coefficients` output, the rank
of the signal's low-rank representation (upstream's `NB_signal`).
"""
rank(m::Model) = m.rank
rank(e::Engine) = rank(e.model)

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

# A target's name, or an ArgumentError.
checktarget(target::Symbol) = target in TARGETS || throw(ArgumentError(
    "target must be :signal_and_derivatives, :signal or :coefficients, not $(repr(target))"))

"""
    outputs(engine, N; target = :signal_and_derivatives)

Fresh output arrays for a batch of `N` voxels, for one target:

- `:signal_and_derivatives`: `(; signal, ∂T₁, ∂T₂)`, each an `(NTR, N)`
  `CuMatrix{ComplexF32}`;
- `:signal`: `(; signal)`, an `(NTR, N)` `CuMatrix{ComplexF32}`;
- `:coefficients`: `(; coefficients)`, a `(rank, N)` `CuMatrix{Float32}`.
"""
function outputs(engine::Engine, N::Integer; target::Symbol = :signal_and_derivatives)
    checktarget(target)
    m = engine.model
    target === :coefficients && return (; coefficients = CuMatrix{Float32}(undef, m.rank, N))
    new() = CuMatrix{ComplexF32}(undef, m.ntr, N)
    return target === :signal ? (; signal = new()) : (signal = new(), ∂T₁ = new(), ∂T₂ = new())
end

"""
    predict!(out, engine, p) -> out

Predict, for every voxel of `p`, what `out` asks for, writing it into `out`
in place: the signal and its derivatives, the signal alone, or the signal's
basis coefficients. Each comes from its own engine, which computes only that.

- `p`: anything with properties `T₁`, `T₂`, `B₁` holding GPU vectors of
  Float32 of equal length N, in physical units (T₁ and T₂ in seconds, B₁
  relative), e.g. MRSTAT's StructArray as it is. Vectors may be strided
  views (positive stride).
- `out`, one of:
  - `(; signal, ∂T₁, ∂T₂)` in any order, each an `(NTR, N)`
    `CuMatrix{ComplexF32}` indexed `[readout, voxel]`: the signal at unit
    proton density, ∂T₁ and ∂T₂ in physical units (per second), not scaled;
  - `(; signal)`, the same signal;
  - `(; coefficients)`, a `(rank, N)` `CuMatrix{Float32}` indexed
    `[coefficient, voxel]`: the signal's coordinates in the model's low-rank
    basis (see [`NeuralBloch.rank`](@ref)), which the bundle does not ship.
  The caller owns them: any address with TensorRT's 256-byte alignment
  (whole arrays have it; see the error for column views), no two
  overlapping.
- `N` must be in [`batchprofile(engine)`](@ref), or 0 (then nothing happens).

Asynchronous: everything is queued on the calling task's stream and ready
in stream order, without waiting for the GPU (so it does not check the
training domain; see [`checkdomain`](@ref)). One exception: when the
previous call for the same output set came from another task, this one
first waits for that task's stream, since that engine's buffers change
hands. Keep `out` and `p` alive until the results are used. Every contract
violation is an `ArgumentError`, thrown before anything is queued. After
warm-up it allocates nothing, on the host or the GPU.
"""
function predict!(out, engine::Engine, p)
    rows, E = outputlayout(out, engine.model.ntr, engine.model.rank)
    N = checkcall(engine, p, alignment(rows, E))
    checkoutputs(out, N, rows, E)
    N == 0 && return out
    t = targetof(engine, out)
    h, io = t.handle, t.io
    @lock engine.lock begin
        stream = CUDACore.stream()
        TensorRT.claim!(h, stream)
        stage!(t.input, p, N, stream)
        TensorRT.set_shape!(h, io.x, (N, 3))
        TensorRT.set_address!(h, io.x, devptr(t.input))
        bind!(h, io, out)
        TensorRT.run!(h, stream)
    end
    return out
end

"""
    predict(engine, p; target = :signal_and_derivatives)

[`predict!`](@ref) into fresh [`outputs`](@ref) for `target`:
`:signal_and_derivatives`, `:signal` or `:coefficients`.
"""
function predict(engine::Engine, p; target::Symbol = :signal_and_derivatives)
    checktarget(target)
    m = engine.model
    step = target === :coefficients ? alignment(m.rank, Float32) : alignment(m.ntr, ComplexF32)
    return predict!(outputs(engine, checkcall(engine, p, step); target), engine, p)
end

# The batch size of a call on `engine` with inputs `p`, once both are known to
# be fine: before outputs are allocated for it, which a batch too large for
# the engine could fail at. `step` is the outputs' alignment, for the advice.
function checkcall(engine::Engine, p, step::Integer)
    isopen(engine) || throw(InvalidStateException("the engine is closed", :closed))
    N = checkinputs(p)
    problem = batch_problem(engine.batch, N, step)
    problem === nothing || throw(ArgumentError(problem))
    return N
end

# The engine that computes an output set (outputkeys_ok has checked it).
targetof(e::Engine, ::NamedTuple{(:signal,)}) = e.targets.signal
targetof(e::Engine, ::NamedTuple{(:coefficients,)}) = e.targets.coefficients
targetof(e::Engine, ::NamedTuple) = e.targets.signal_and_derivatives    # (; signal, ∂T₁, ∂T₂), any order

# Bind every output to the engine's tensor of the same name. One method per
# output set, with the names spelled out: mapping over the names instead
# allocates for the three outputs.
bind!(h, io, out::NamedTuple{(:signal,)}) = TensorRT.set_address!(h, io.signal, devptr(out.signal))
bind!(h, io, out::NamedTuple{(:coefficients,)}) = TensorRT.set_address!(h, io.coefficients, devptr(out.coefficients))
function bind!(h, io, out::NamedTuple)                                   # (; signal, ∂T₁, ∂T₂), any order
    TensorRT.set_address!(h, io.signal, devptr(out.signal))
    TensorRT.set_address!(h, io.∂T₁, devptr(out.∂T₁))
    TensorRT.set_address!(h, io.∂T₂, devptr(out.∂T₂))
end

# Taking the pointer hands the array to the task's stream (CUDA.jl orders
# its free after our work); TensorRT then gets the bare address.
devptr(A::CuArray) = reinterpret(Ptr{Cvoid}, pointer(A))

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
        "$(m.family) v$(m.version) in $(m.dir) (NTR $(m.ntr), rank $(m.rank), $(m.provenance["upstream"]))"
    catch err
        err isa BundleError || rethrow()
        sprint(showerror, err)
    end
    println(io, "  model: ", model)
    println(io, "  cache: ", TensorRT.cache_root(), "\n")
    TensorRT.versioninfo(io)
end
