# Running engines: a Handle is one engine with one execution context, driven
# through libneuralbloch.so (engine.cpp). Shapes and addresses are set per
# run, so outputs can live anywhere; nothing here allocates on the hot path.

# TensorRT's DataType enum → Julia element type (types Julia lacks map to nothing).
const ELTYPES = Dict{Int32,DataType}(0 => Float32, 1 => Float16, 2 => Int8, 3 => Int32, 4 => Bool,
                                     5 => UInt8, 8 => Int64)

# libneuralbloch.so's entry points, looked up once per library.
struct API
    last_error::Ptr{Cvoid}
    open::Ptr{Cvoid}
    close::Ptr{Cvoid}
    io_count::Ptr{Cvoid}
    io_name::Ptr{Cvoid}
    io_is_input::Ptr{Cvoid}
    io_dtype::Ptr{Cvoid}
    io_shape::Ptr{Cvoid}
    io_profile::Ptr{Cvoid}
    device_memory::Ptr{Cvoid}
    set_shape::Ptr{Cvoid}
    set_address::Ptr{Cvoid}
    enqueue::Ptr{Cvoid}
end

API(lib::Ptr{Cvoid}) = API((Libdl.dlsym(lib, Symbol(:nb_, f)) for f in fieldnames(API))...)

const APIS = Dict{String,API}()
const APIS_LOCK = ReentrantLock()

api(library::AbstractString) = @lock APIS_LOCK get!(() -> API(Libdl.dlopen(library)), APIS, library)

lasterror(api::API) = unsafe_string(ccall(api.last_error, Cstring, ()))

"""
    TensorRTError(msg)

TensorRT refused something at run time; the message carries its own words.
"""
struct TensorRTError <: Exception
    msg::String
end

Base.showerror(io::IO, e::TensorRTError) = print(io, "TensorRTError: ", e.msg)

"""
    TensorInfo

One input or output of an engine: `name`, whether it is an `input`, its
Julia `eltype` (`nothing` when Julia has no such type), its `shape` in
TensorRT's (row-major) order with -1 for a dynamic dimension, and for an
input the `profile`: its shapes at MIN, OPT and MAX (empty for outputs).
"""
struct TensorInfo
    name::String
    input::Bool
    eltype::Union{DataType,Nothing}
    shape::Vector{Int}
    profile::NTuple{3,Vector{Int}}
end

"""
    Handle(path; native = compile())

The serialized engine in `path`, opened by TensorRT with one execution
context, in the calling task's CUDA context. The context allocates the
engine's scratch memory now (see [`scratch_bytes`](@ref)), outside CUDA.jl's
pool.

Per run: [`claim!`](@ref) the handle for the stream, [`set_shape!`](@ref)
the inputs, [`set_address!`](@ref) every input and output, [`run!`](@ref).
Not thread-safe: one run is set up at a time, and nothing may use `h` while
another thread closes it (the Engine's lock sees to both). `close` releases
it early, from any thread, any number of times; otherwise a finalizer does.
"""
mutable struct Handle
    @atomic ptr::Ptr{Cvoid}                    # C_NULL once closed; swapped out exactly once
    const api::API
    const path::String
    const context::CUDACore.CuContext
    const tensors::Vector{TensorInfo}
    const dims::Vector{Int64}                  # shape buffer for set_shape!
    stream::Union{Nothing,CUDACore.CuStream}   # of the latest claim!
    event::Union{Nothing,CUDACore.CuEvent}
end

function Handle(path::AbstractString; native::Native = compile())
    a = api(native.library)
    context = CUDACore.context()
    CUDACore.activate(context)
    ptr = ccall(a.open, Ptr{Cvoid}, (Cstring,), path)
    if ptr == C_NULL
        # Creating the context allocates its scratch memory eagerly, and
        # memory held by CUDA.jl's pool may be what it is missing.
        CUDACore.reclaim()
        ptr = ccall(a.open, Ptr{Cvoid}, (Cstring,), path)
    end
    ptr == C_NULL && throw(TensorRTError("cannot open the TensorRT engine: " * lasterror(a)))
    h = Handle(ptr, a, path, context, TensorInfo[], zeros(Int64, 8), nothing, nothing)
    finalizer(release!, h)
    n = ccall(a.io_count, Int32, (Ptr{Cvoid},), ptr)
    append!(h.tensors, (tensorinfo(h, i) for i in 0:n-1))
    return h
end

function tensorinfo(h::Handle, i::Integer)
    a, p, d = h.api, h.ptr, h.dims
    name = unsafe_string(ccall(a.io_name, Cstring, (Ptr{Cvoid}, Int32), p, i))
    input = ccall(a.io_is_input, Int32, (Ptr{Cvoid}, Int32), p, i) == 1
    eltype = get(ELTYPES, ccall(a.io_dtype, Int32, (Ptr{Cvoid}, Int32), p, i), nothing)
    dims(rank) = Int[d[k] for k in 1:rank]
    shape = dims(ccall(a.io_shape, Int32, (Ptr{Cvoid}, Int32, Ptr{Int64}), p, i, d))
    profile = ntuple(3) do which
        input ? dims(ccall(a.io_profile, Int32, (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}), p, i, which - 1, d)) : Int[]
    end
    return TensorInfo(name, input, eltype, shape, profile)
end

# Free the TensorRT objects exactly once, however many threads get here: the
# pointer is swapped out before it is used. Without making the context
# current, since this is also the finalizer's path: TensorRT's CUDA runtime
# then falls back to device 0's primary context, CUDA.jl's own.
function release!(h::Handle)
    ptr = @atomicswap h.ptr = C_NULL
    ptr == C_NULL || ccall(h.api.close, Cvoid, (Ptr{Cvoid},), ptr)
    return nothing
end

"""
    close(h::Handle)

Release the engine, its execution context and scratch memory, after the runs
queued on the latest claimed stream have finished. Safe to call more than
once, from any thread; any later use of `h` throws.
"""
function Base.close(h::Handle)
    isopen(h) || return nothing
    stream = h.stream
    stream === nothing || CUDACore.synchronize(stream)   # queued runs use the scratch memory
    CUDACore.activate(h.context)
    return release!(h)
end

Base.isopen(h::Handle) = h.ptr != C_NULL

@inline function checkopen(h::Handle)
    ptr = h.ptr
    ptr == C_NULL && throw_closed(h)
    return ptr
end

@noinline throw_closed(h::Handle) = throw(InvalidStateException("the TensorRT engine $(h.path) is closed", :closed))

"""
    tensors(h) -> Vector{TensorInfo}

The engine's inputs and outputs, by index: `tensors(h)[i + 1]` is tensor `i`.
"""
tensors(h::Handle) = h.tensors

"""
    tensorindex(h, name) -> Union{Int,Nothing}

The index of the tensor called `name`, or `nothing`.
"""
function tensorindex(h::Handle, name::AbstractString)
    i = findfirst(t -> t.name == name, h.tensors)
    return i === nothing ? nothing : i - 1
end

"""
    scratch_bytes(h) -> Int

Bytes of GPU scratch memory the execution context holds.
"""
scratch_bytes(h::Handle) = Int(ccall(h.api.device_memory, Int64, (Ptr{Cvoid},), checkopen(h)))

"""
    set_shape!(h, i, dims)

Set the shape of input `i` for the next runs: `dims` in TensorRT's order. An
`ArgumentError` names the range the engine's optimization profile accepts
when `dims` is outside it (checked before TensorRT sees them).
"""
function set_shape!(h::Handle, i::Integer, dims)
    p = checkopen(h)
    t = h.tensors[i+1]
    t.input || throw(ArgumentError("$(t.name) is an output of the engine; only inputs have a shape to set"))
    lo, _, hi = t.profile
    inside = length(dims) == length(lo)
    for k in eachindex(lo)
        inside || break
        inside = lo[k] ≤ dims[k] ≤ hi[k]
    end
    inside || throw_profile(t, dims)
    for k in eachindex(dims)
        h.dims[k] = dims[k]
    end
    rc = ccall(h.api.set_shape, Int32, (Ptr{Cvoid}, Int32, Ptr{Int64}, Int32), p, i, h.dims, length(dims))
    rc == 0 || throw(TensorRTError(lasterror(h.api)))
    return h
end

@noinline throw_profile(t::TensorInfo, dims) = throw(ArgumentError(
    "shape $(Tuple(dims)) of input $(t.name) is outside the engine's optimization profile, " *
    "which accepts $(Tuple(t.profile[1])) to $(Tuple(t.profile[3]))"))

"""
    set_address!(h, i, pointer)

Bind tensor `i` to device memory at `pointer` for the next runs. TensorRT
requires 256-byte alignment (dev/NOTES.md, S6); callers check it.
"""
function set_address!(h::Handle, i::Integer, pointer::Ptr{Cvoid})
    rc = ccall(h.api.set_address, Int32, (Ptr{Cvoid}, Int32, Ptr{Cvoid}), checkopen(h), i, pointer)
    rc == 0 || throw(TensorRTError(lasterror(h.api)))
    return h
end

"""
    claim!(h, stream = CUDACore.stream())

Order everything queued on `stream` from now on after all earlier runs of
`h`, on whichever stream those were queued. All runs share the execution
context's scratch memory, and callers usually share input buffers too, so
two runs queued on different streams must not overlap on the GPU. Call it
before queuing anything a run uses; it costs nothing while the stream stays
the same.
"""
function claim!(h::Handle, stream::CUDACore.CuStream = CUDACore.stream())
    checkopen(h)
    last = h.stream
    last === stream && return h
    if last !== nothing
        e = h.event
        if e === nothing
            e = h.event = CUDACore.CuEvent(CUDACore.EVENT_DISABLE_TIMING)
        end
        CUDACore.record(e, last)
        CUDACore.wait(e, stream)
    end
    h.stream = stream
    return h
end

"""
    run!(h, stream = CUDACore.stream())

Queue one run of the engine on `stream`, with the shapes and addresses set.
Asynchronous: the outputs are ready in stream order.
"""
function run!(h::Handle, stream::CUDACore.CuStream = CUDACore.stream())
    p = checkopen(h)
    CUDACore.activate(h.context)                 # TensorRT runs in the thread's current context
    rc = ccall(h.api.enqueue, Int32, (Ptr{Cvoid}, CUDACore.CUstream), p, stream)
    rc == 0 || throw(TensorRTError(lasterror(h.api)))
    return nothing
end
