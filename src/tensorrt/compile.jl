# Compiling csrc/ with g++ into the cache: libneuralbloch.so from engine.cpp
# and the neuralbloch-build executable from builder.cpp, once per source
# content, TensorRT version and compiler.

const LIBRARY = "libneuralbloch.so"
const BUILDER = "neuralbloch-build"

"""
    Native(dir, csrc)

The compiled products in their cache entry `dir`: the runtime library
(`library`) and the builder executable (`builder`), compiled from sources
with digest `csrc` (see [`csrc_hash`](@ref)).
"""
struct Native
    dir::String
    library::String
    builder::String
    csrc::String
end

Native(dir::AbstractString, csrc::AbstractString) = Native(dir, joinpath(dir, LIBRARY), joinpath(dir, BUILDER), csrc)

"""
    CompileError(cmd, output, tensorrt)

g++ failed; `output` is everything it printed, shown verbatim.
"""
struct CompileError <: Exception
    cmd::Cmd
    output::String
    tensorrt::String
end

function Base.showerror(io::IO, e::CompileError)
    println(io, "CompileError: g++ could not compile NeuralBloch's C++ code against TensorRT ",
            e.tensorrt, ". NeuralBloch is tested with TensorRT 11.2; another release may have ",
            "changed an API it uses. g++ said:\n")
    println(io, rstrip(e.output))
    print(io, "\ncommand: ", sprint(show, e.cmd))    # plain text, even on a terminal
end

"""
    csrc_dir() -> String

The package's C++ sources.
"""
csrc_dir() = joinpath(pkgdir(parentmodule(@__MODULE__)), "csrc")

"""
    csrc_hash(dir = csrc_dir()) -> String

The identity of the C++ sources: a digest of every file's name and content
(not timestamps).
"""
csrc_hash(dir::AbstractString = csrc_dir()) =
    digest(Dict(name => bytes2hex(SHA.sha256(read(joinpath(dir, name))))
                for name in readdir(dir) if isfile(joinpath(dir, name))))

native_key(sys::System, csrc) = Dict{String,Any}(
    "csrc_sha256" => csrc_hash(csrc),
    "tensorrt" => vstring(sys.nvinfer.version),
    "tensorrt_headers" => vstring(sys.headers.version),
    "compiler" => "g++ $(sys.compiler.version)",
    "arch" => string(Sys.ARCH),
)

# The directories g++ searches by itself. Passing one of them again with
# -isystem reorders the system directories, which breaks libstdc++'s
# #include_next.
default_includes() = ("/usr/include", "/usr/include/$(Sys.ARCH)-linux-gnu")

# Compile `src` into `dir/name`; return the command, or throw a CompileError.
# g++ runs in `dir`, so the command names its output relative to the entry.
#
# Deliberately no -static-libstdc++: Julia dlopens with RTLD_DEEPBIND, and a
# second, static C++ runtime inside the library would shadow the process's
# libstdc++ with its own uninitialized state; libnvinfer's static
# initializers then crash (MRSTAT.jl's shim found this out).
function gxx(sys::System, src, dir, name; shared::Bool)
    includes = [["-isystem", d] for d in (sys.headers.dir, sys.cuda.dir) if !(d in default_includes())]
    major = sys.nvinfer.version.major
    libs = ["-l:libnvinfer.so.$major"; shared ? String[] : ["-l:libnvonnxparser.so.$major"]]
    cmd = `$(sys.compiler.path) -std=c++17 -O2 -Wall -Wextra -fvisibility=hidden
           -fvisibility-inlines-hidden $(shared ? ["-fPIC", "-shared"] : String[])
           $(reduce(vcat, includes; init = String[])) $src -o $name
           -L$(dirname(sys.nvinfer.path)) $libs -Wl,--no-undefined`
    output = IOBuffer()
    p = run(pipeline(ignorestatus(Cmd(cmd; dir)); stdout = output, stderr = output))
    success(p) || throw(CompileError(cmd, String(take!(output)), vstring(sys.nvinfer.version)))
    return cmd
end

"""
    compile([sys]; csrc = csrc_dir(), root) -> Native

`libneuralbloch.so` and `neuralbloch-build`, compiled from `csrc` against
the system's TensorRT with g++: from the cache (`root`, by default in
[`cache_root`](@ref)) when the same sources were compiled before with the
same TensorRT and compiler. Throws a [`PrerequisiteError`](@ref) when the
system lacks something compiling needs, and a [`CompileError`](@ref),
carrying g++'s output, when compiling fails.
"""
function compile(sys::System = detect(); csrc::AbstractString = csrc_dir(),
                 root::AbstractString = joinpath(cache_root(), "native"))
    require(sys; gpu = false)
    key = native_key(sys, csrc)
    dir = getorbuild(root, key; what = "NeuralBloch's C++ code") do tmp
        @info "Compiling NeuralBloch's C++ code against TensorRT $(vstring(sys.nvinfer.version)) " *
              "(once per TensorRT version and source change)."
        t = time()
        library = gxx(sys, joinpath(csrc, "engine.cpp"), tmp, LIBRARY; shared = true)
        builder = gxx(sys, joinpath(csrc, "builder.cpp"), tmp, BUILDER; shared = false)
        return Dict{String,Any}("library_command" => sprint(show, library),
                                "builder_command" => sprint(show, builder),
                                "seconds" => round(time() - t; digits = 2))
    end
    return Native(dir, key["csrc_sha256"])
end
