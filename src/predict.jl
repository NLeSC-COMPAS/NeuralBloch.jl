# L2: the Engine and predict!. The Engine holds the TensorRT engines of the
# model on this GPU, one engine for each target.

"""
    Target

The engine of one target: its TensorRT handle, the 0-based indices of its
tensors (`x` and the outputs of the target, by public name), and its own input
buffer.

Each target has its own buffer because `TensorRT.claim!` puts the runs of one
handle in order, but not the runs of different handles. With one shared
buffer, a run on the stream of one task could overwrite the input of a run of
a different target on a different stream, before that run reads it.
"""
struct Target{IO<:NamedTuple}
    handle::TensorRT.Handle
    io::IO
    # A concrete type: CuMatrix{T} does not set the memory type. An abstract
    # field makes each call on it dynamic, and dynamic calls allocate.
    input::CuArray{Float32,2,DeviceMemory}       # (3, max): [T₁; T₂; B₁] per voxel, the engine's x
end

const Targets = @NamedTuple{signal_and_derivatives::Target{@NamedTuple{x::Int, signal::Int, ∂T₁::Int, ∂T₂::Int}},
                            signal::Target{@NamedTuple{x::Int, signal::Int}},
                            coefficients::Target{@NamedTuple{x::Int, coefficients::Int}}}

"""
    Engine(model = Model(); batch = (1, 14400, 40000), tf32 = false)

The TensorRT engines of the model for the GPU of the calling task, one engine
for each target, ready for [`predict!`](@ref). The targets are the signal and
its derivatives, the signal alone, and the basis coefficients of the signal.
Each engine calculates only what its target needs.

The first time on a machine, and after a change of TensorRT, GPU or settings,
`Engine` compiles the C++ code of NeuralBloch against the system TensorRT. It
then builds the three engines in subprocesses, in some seconds each, and
tells you. The cache keeps all of them, and later calls load them from the
cache.

- `batch = (min, opt, max)`: the batch sizes that the engines accept, and the
  size that they are tuned for. The execution context of each engine holds GPU
  scratch memory in proportion to `max`: approximately 11 KiB for each voxel
  for the signal and its derivatives (440 MiB at the default), and less for
  the other targets (see `dev/NOTES.md`). A batch outside the range causes an
  error. The engines do not pad batches.
- `tf32`: allow TF32 inner products (Ampere and newer). The default is off,
  because accuracy is more important.

`Engine` throws a `PrerequisiteError` if the system does not have a necessary
item (see [`NeuralBloch.versioninfo`](@ref)), a `BuildError` if a build
fails, and a [`BundleError`](@ref) if an engine does not agree with the model
card. `close` releases the GPU memory of the engines. If you do not call it,
the garbage collector releases the memory.
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
    # Build all the engines before you open one: an open execution context
    # holds GPU memory that the builder subprocess can need. (build first checks
    # all the prerequisites, the GPU included, and compiles what it needs.)
    paths = map(target -> TensorRT.build(model.onnx[target]; settings, sys), TARGETS)
    native = TensorRT.compile(sys)
    handles = TensorRT.Handle[]
    try
        for path in paths
            push!(handles, TensorRT.Handle(path; native))
        end
        ios = map((target, h) -> resolve_io(TensorRT.tensors(h), target; model.ntr, model.rank), TARGETS, handles)
        # One `settings` built all the engines (and is part of their cache key),
        # thus each engine must have its batch profile.
        for (target, path, h, io) in zip(TARGETS, paths, handles, ios)
            profile = Tuple(shape[1] for shape in TensorRT.tensors(h)[io.x+1].profile)
            profile == settings.batch || throw(TensorRT.BuildError(
                "the cached $target engine $path has the batch profile $profile, not $(settings.batch); " *
                "the cache is not correct: remove $(dirname(path)) and call Engine again"))
        end
        lo, opt, hi = settings.batch
        targets = NamedTuple{TARGETS}(map((h, io) -> Target(h, io, CuMatrix{Float32}(undef, 3, hi)), handles, ios))
        return Engine(model, targets, (min = lo, opt, max = hi), sys.gpu.name, ReentrantLock())
    catch
        foreach(close, handles)
        rethrow()
    end
end

Base.show(io::IO, e::Engine) = print(io, "Engine(NTR ", e.model.ntr, ", rank ", e.model.rank, ", batch ", e.batch.min,
                                     "–", e.batch.max, " (opt ", e.batch.opt, "), ", e.gpu,
                                     isopen(e) ? "" : ", closed", ")")

"""
    close(engine::Engine)

Release the GPU memory of the engines, of TensorRT and of their own buffers,
after the runs in the queue are complete. You can call it more than one time
and from any task. A `predict!` call after `close`, or a call that waits for
`close`, throws an `InvalidStateException`.
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
Base.isopen(e::Engine) = isopen(first(e.targets).handle)     # all the engines close together

# --- metadata -----------------------------------------------------------------------

"""
    ntr(model_or_engine) -> Int

The readouts for each voxel: the rows of the signal and its derivatives.
"""
ntr(m::Model) = m.ntr
ntr(e::Engine) = ntr(e.model)

"""
    rank(model_or_engine) -> Int

The basis coefficients for each voxel: the rows of the `coefficients` output.
This is the rank of the low-rank representation of the signal (`NB_signal`
upstream).
"""
rank(m::Model) = m.rank
rank(e::Engine) = rank(e.model)

"""
    domain(model_or_engine) -> (; T₁, T₂, B₁)

The training domain, `(low, high)` for each property: T₁ and T₂ in seconds,
B₁ relative. The training data also had only T₂ < T₁. Outside the domain, the
network extrapolates. See [`checkdomain`](@ref).
"""
domain(m::Model) = NamedTuple{FEATURES}(map(name -> m.domain[Symbol(cardname(name))], FEATURES))
domain(e::Engine) = domain(e.model)

"""
    batchprofile(engine) -> (; min, opt, max)

The batch sizes that `engine` accepts (`min` to `max` voxels in each call),
and the size that it is tuned for.
"""
batchprofile(e::Engine) = e.batch

"""
    sequence(model_or_engine) -> NamedTuple

The FISP3D sequence of the simulation that made the training data: TR, TE,
TI, TW (seconds), max_state, repetitions, inversion_prepulse, wait_spoiling,
undersampling_factor, delta_k_spoil (rad/m), and the RF train as rf_angle and
rf_phase (degrees, one value for each readout).
"""
sequence(m::Model) = m.sequence
sequence(e::Engine) = sequence(e.model)

# --- predicting ---------------------------------------------------------------------

"""
    outputs(engine, N; target = :signal_and_derivatives)

New output arrays for a batch of `N` voxels, for one target:

- `:signal_and_derivatives`: `(; signal, ∂T₁, ∂T₂)`, each an `(NTR, N)`
  `CuMatrix{ComplexF32}`;
- `:signal`: `(; signal)`, an `(NTR, N)` `CuMatrix{ComplexF32}`;
- `:coefficients`: `(; coefficients)`, a `(rank, N)` `CuMatrix{Float32}`.
"""
function outputs(engine::Engine, N::Integer; target::Symbol = :signal_and_derivatives)
    rows, E = outputlayout(target, engine.model.ntr, engine.model.rank)
    names = map(publicname, OUTPUTS[target])
    return NamedTuple{names}(map(_ -> CuMatrix{E}(undef, rows, N), names))
end

"""
    predict!(out, engine, p) -> out

For each voxel of `p`, predict what `out` asks for, and write it into `out`.
The possible outputs are the signal and its derivatives, the signal alone, or
the basis coefficients of the signal. Each output set comes from its own
engine, which calculates only that set.

- `p`: an object with the properties `T₁`, `T₂` and `B₁`. Each property is a
  GPU vector of Float32 of length N, in physical units (T₁ and T₂ in seconds,
  B₁ relative). For example, you can use the StructArray of MRSTAT without
  changes. A vector can be a strided view with a positive stride.
- `out`, one of these sets:
  - `(; signal, ∂T₁, ∂T₂)` in any order. Each is an `(NTR, N)`
    `CuMatrix{ComplexF32}`, indexed `[readout, voxel]`: the signal at unit
    proton density, and ∂T₁ and ∂T₂ in physical units (per second), not
    scaled.
  - `(; signal)`: the same signal.
  - `(; coefficients)`: a `(rank, N)` `CuMatrix{Float32}`, indexed
    `[coefficient, voxel]`. These are the coordinates of the signal in the
    low-rank basis of the model (see [`NeuralBloch.rank`](@ref)). The bundle
    does not include the basis.
  The caller owns the outputs. Each output must start at an address with the
  256-byte alignment of TensorRT (whole arrays have it; for column views, see
  the error message). Two outputs must not overlap.
- `N` must be in [`batchprofile(engine)`](@ref), or 0. If `N` is 0, the
  function does nothing.

`predict!` is asynchronous. It puts all the work in the queue of the stream of
the calling task, and does not wait for the GPU. The results are ready in
stream order. Thus, it does not check the training domain (see
[`checkdomain`](@ref)). One exception: if the previous call for the same
output set came from a different task, this call first waits for the stream of
that task, because the buffers of that engine go to a different task. Keep
`out` and `p` until you use the results. Each violation of the contract
causes an `ArgumentError` before the function puts work in the queue. After
the warm-up, `predict!` does not allocate memory on the host or on the GPU.
"""
function predict!(out, engine::Engine, p)
    target = targetof(out)
    rows, E = outputlayout(target, engine.model.ntr, engine.model.rank)
    N = checkcall(engine, p, alignment(rows, E))
    checkoutputs(out, N, rows, E)
    N == 0 && return out
    t = engine.targets[target]
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

[`predict!`](@ref) into new [`outputs`](@ref) for `target`:
`:signal_and_derivatives`, `:signal` or `:coefficients`.
"""
function predict(engine::Engine, p; target::Symbol = :signal_and_derivatives)
    m = engine.model
    N = checkcall(engine, p, alignment(outputlayout(target, m.ntr, m.rank)...))
    return predict!(outputs(engine, N; target), engine, p)
end

# Return the batch size of a call on `engine` with the inputs `p`, after the
# checks of the engine and the inputs. Do these checks before you allocate the
# outputs: the allocation can fail for a batch that is too large for the
# engine. `step` is the alignment of the outputs, for the advice in the error.
function checkcall(engine::Engine, p, step::Integer)
    isopen(engine) || throw(InvalidStateException("the engine is closed", :closed))
    N = checkinputs(p)
    problem = batch_problem(engine.batch, N, step)
    problem === nothing || throw(ArgumentError(problem))
    return N
end

# Bind each output to the engine tensor with the same name. There is one
# method for each output set, with the names written out. A map over the names
# allocates for the set of three outputs.
bind!(h, io, out::NamedTuple{(:signal,)}) = TensorRT.set_address!(h, io.signal, devptr(out.signal))
bind!(h, io, out::NamedTuple{(:coefficients,)}) = TensorRT.set_address!(h, io.coefficients, devptr(out.coefficients))
function bind!(h, io, out::NamedTuple)                                   # (; signal, ∂T₁, ∂T₂), any order
    TensorRT.set_address!(h, io.signal, devptr(out.signal))
    TensorRT.set_address!(h, io.∂T₁, devptr(out.∂T₁))
    TensorRT.set_address!(h, io.∂T₂, devptr(out.∂T₂))
end

# The pointer gives the array to the stream of the task: CUDA.jl releases the
# array only after the work on that stream. TensorRT then gets the address.
devptr(A::CuArray) = reinterpret(Ptr{Cvoid}, pointer(A))

# Copy the vectors of p into the rows of the input buffer, on `stream`. These
# are three strided copies from device to device, which do not allocate
# (dev/NOTES.md, S10). Row k of the (3, max) buffer is each third Float32.
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

Throw an `ArgumentError` if a voxel of `p` is not in the training domain
([`domain`](@ref)). Each of `p.T₁`, `p.T₂` and `p.B₁` that `p` has must be in
its range (NaN is not in the range), and they must have the same length. If
`p` has T₁ and T₂, T₂ < T₁ must be true. Outside the domain, the network extrapolates. If `p` has no voxels
(empty vectors), no voxel is outside the domain.

`p` can hold GPU or CPU vectors. For GPU vectors, the function copies
reductions to the host, thus it synchronizes. For this reason,
[`predict!`](@ref) does not call it.
"""
function checkdomain(model::Model, p)
    present = filter(name -> hasproperty(p, name), FEATURES)
    isempty(present) && throw(ArgumentError("p has none of the properties T₁, T₂, B₁ that checkdomain checks"))
    lengths = map(name -> length(getproperty(p, name)), present)
    allequal(lengths) || throw(ArgumentError(
        "$(join(("p.$name" for name in present), ", ", " and ")) have lengths $(join(lengths, ", ", " and ")); " *
        "they must be equal"))
    for name in present
        v = getproperty(p, name)
        isempty(v) && continue                   # extrema throws for an empty CPU vector
        lo, hi = model.domain[Symbol(cardname(name))]
        vmin, vmax = extrema(v)
        within(eltype(v), lo, hi, vmin, vmax) && continue
        throw(ArgumentError("p.$name ranges over [$vmin, $vmax], outside the training domain [$lo, $hi]; " *
                            "the network extrapolates there"))
    end
    if model.constraint == "T2 < T1"
        if hasproperty(p, :T₁) && hasproperty(p, :T₂)
            n = count(p.T₂ .>= p.T₁)
            n == 0 || throw(ArgumentError(
                "$n voxels have T₂ ≥ T₁, outside the training domain (T₂ < T₁ only); the network extrapolates there"))
        end
    else                                         # Model accepts only the constraints in CONSTRAINTS
        error("checkdomain does not check the domain constraint $(repr(model.constraint))")
    end
    return p
end

checkdomain(engine::Engine, p) = checkdomain(engine.model, p)

# [vmin, vmax] ⊆ [lo, hi], with the bounds in the precision of the inputs,
# rounded outward. Thus, a bound rounded to Float32 is in the range: 0.01f0 <
# 0.01, but 0.01f0 is the bound as MRSTAT stores it.
within(::Type{T}, lo, hi, vmin, vmax) where {T<:AbstractFloat} = T(lo, RoundDown) ≤ vmin && vmax ≤ T(hi, RoundUp)
within(::Type, lo, hi, vmin, vmax) = lo ≤ vmin && vmax ≤ hi

# --- reporting ----------------------------------------------------------------------

"""
    NeuralBloch.versioninfo([io])

Show what NeuralBloch uses on this machine: the model, the cache, and the
system prerequisites (the TensorRT libraries and headers, the CUDA headers,
g++, the GPU). For each missing item, it shows the apt or dnf command that
installs it. If the model is not installed or not correct, the report shows
the error and continues.
"""
function versioninfo(io::IO = stdout)
    println(io, "NeuralBloch ", pkgversion(@__MODULE__))
    # The report continues without the model: a bundle that is not correct or
    # not installed (Artifacts throws an ErrorException) is a problem to show.
    model = try
        m = Model()
        "$(m.family) v$(m.version) in $(m.dir) (NTR $(m.ntr), rank $(m.rank), $(m.provenance["upstream"]))"
    catch err
        err isa InterruptException && rethrow()
        sprint(showerror, err)
    end
    println(io, "  model: ", model)
    println(io, "  cache: ", TensorRT.cache_root(), "\n")
    TensorRT.versioninfo(io)
end
