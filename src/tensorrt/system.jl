# What this machine provides for building and running TensorRT engines.
#
# Everything comes from NVIDIA's deb/rpm packages (dev/NOTES.md, S1): the
# runtime libraries, TensorRT's headers, the CUDA runtime headers those
# headers include, and g++. This file only looks for them and says how to
# install what is missing; it never installs or downloads anything.

"""
    Library(path, version)

A system shared library, loaded into this process, and the version it reports.
"""
struct Library
    path::String
    version::VersionNumber
end

"""
    Headers(dir, version)

The directory holding TensorRT's headers, and the version `NvInferVersion.h` states.
"""
struct Headers
    dir::String
    version::VersionNumber
end

"""
    CUDAHeaders(dir, complete)

The directory holding `cuda_runtime_api.h`, which TensorRT's headers include.
`complete` is false when `crt/host_defines.h`, which it includes in turn, is
missing: the two come from different packages.
"""
struct CUDAHeaders
    dir::String
    complete::Bool
end

"""
    Compiler(path, version)
"""
struct Compiler
    path::String
    version::VersionNumber
end

"""
    GPU(name, capability, uuid)

The CUDA device of the calling task.
"""
struct GPU
    name::String
    capability::VersionNumber
    uuid::Base.UUID
end

"""
    System

What [`detect`](@ref) found; `nothing` marks a missing piece. For the GPU,
`gpu_error` says why it is missing.
"""
struct System
    nvinfer::Union{Library,Nothing}
    onnxparser::Union{Library,Nothing}
    headers::Union{Headers,Nothing}
    cuda::Union{CUDAHeaders,Nothing}
    compiler::Union{Compiler,Nothing}
    gpu::Union{GPU,Nothing}
    gpu_error::String
end

# --- parsing -------------------------------------------------------------------

"""
    vstring(v::VersionNumber)

TensorRT's own spelling of a version: the build number as a fourth field
(`11.2.1.2`), as NVIDIA's packages print it.
"""
vstring(v::VersionNumber) = join((v.major, v.minor, v.patch, v.build...), '.')

"""
    parse_header_version(text) -> Union{VersionNumber,Nothing}

The version stated by the text of `NvInferVersion.h`, with the build number as
the version's build field. TensorRT 11 defines the numbers as
`TRT_MAJOR_ENTERPRISE` etc. and aliases `NV_TENSORRT_MAJOR` to them; older
headers define `NV_TENSORRT_MAJOR` etc. directly. Both spellings are accepted.
"""
function parse_header_version(text::AbstractString)
    parts = map(("MAJOR", "MINOR", "PATCH", "BUILD")) do field
        m = match(Regex("#define\\s+(?:TRT_$(field)_ENTERPRISE|NV_TENSORRT_$field)\\s+(\\d+)\\b"), text)
        m === nothing ? nothing : parse(Int, m[1])
    end
    any(isnothing, parts) && return nothing
    return VersionNumber(parts[1], parts[2], parts[3], (), (parts[4],))
end

"""
    parse_compiler_version(text) -> Union{VersionNumber,Nothing}

The version printed by `g++ -dumpfullversion`.
"""
function parse_compiler_version(text::AbstractString)
    m = match(r"^\s*(\d+(?:\.\d+){0,2})\s*$", text)
    return m === nothing ? nothing : VersionNumber(m[1])
end

"""
    parse_package_cuda(version) -> Union{VersionNumber,Nothing}

The CUDA version a TensorRT package was built for, from its package version:
`11.2.1.2-1+cuda13.3` (deb) or `11.2.1.2-1.cuda13.3` (rpm) give `v"13.3"`.
"""
function parse_package_cuda(version::AbstractString)
    m = match(r"[+.]cuda(\d+)\.(\d+)", version)
    return m === nothing ? nothing : VersionNumber(parse(Int, m[1]), parse(Int, m[2]))
end

# --- detection -----------------------------------------------------------------

# The sonames tried, in this order: 11 is the tested release, 10 the oldest
# with the API csrc uses. Older libraries also encode their version
# differently, so they are never loaded.
const MAJORS = (11, 12, 10)

function find_headers()
    for dir in ("/usr/include/$(Sys.ARCH)-linux-gnu", "/usr/include")
        all(f -> isfile(joinpath(dir, f)), ("NvInfer.h", "NvOnnxParser.h", "NvInferVersion.h")) || continue
        v = parse_header_version(read(joinpath(dir, "NvInferVersion.h"), String))
        v === nothing || return Headers(dir, v)
    end
    return nothing
end

# The deb/rpm CUDA packages install into /usr/local/cuda-X.Y/targets/<arch>/,
# with /usr/local/cuda pointing at the default one; distributions' own CUDA
# packages use /usr/include.
function find_cuda_headers()
    targets = Sys.ARCH === :aarch64 ? ("sbsa-linux", "aarch64-linux") : ("$(Sys.ARCH)-linux",)
    versioned = isdir("/usr/local") ? filter(startswith("cuda-"), readdir("/usr/local")) : String[]
    sort!(versioned; by = d -> something(tryparse(VersionNumber, d[6:end]), v"0"), rev = true)
    dirs = ["/usr/local/cuda/include";
            [joinpath("/usr/local", d, "targets", t, "include") for d in versioned for t in targets];
            "/usr/include"]
    found = [CUDAHeaders(d, isfile(joinpath(d, "crt", "host_defines.h")))
             for d in dirs if isfile(joinpath(d, "cuda_runtime_api.h"))]
    isempty(found) && return nothing
    return found[something(findfirst(h -> h.complete, found), 1)]
end

# dlopen the first `stem.so.<major>` that loads; the library stays loaded.
function load_library(stem, majors)
    for m in majors
        h = Libdl.dlopen("$stem.so.$m"; throw_error = false)
        h === nothing || return h
    end
    return nothing
end

# getInferLibVersion() is major·10⁴ + minor·10² + patch since TensorRT 10;
# the build number has its own function.
function nvinfer_version(h)
    v = Int(ccall(Libdl.dlsym(h, :getInferLibVersion), Int32, ()))
    b = Libdl.dlsym(h, :getInferLibBuildVersion; throw_error = false)
    build = b === nothing ? () : (Int(ccall(b, Int32, ())),)
    return VersionNumber(v ÷ 10000, v ÷ 100 % 100, v % 100, (), build)
end

# The ONNX parser has a version scheme of its own (0.2.0 in TensorRT 11.2),
# so it is shown, never compared: the soname major pairs it with libnvinfer.
function parser_version(h)
    v = Int(ccall(Libdl.dlsym(h, :getNvOnnxParserVersion), Cint, ()))
    return VersionNumber(v ÷ 10000, v ÷ 100 % 100, v % 100)
end

function find_compiler()
    path = Sys.which("g++")
    path === nothing && return nothing
    text = try
        readchomp(`$path -dumpfullversion`)
    catch
        return nothing
    end
    v = parse_compiler_version(text)
    return v === nothing ? nothing : Compiler(path, v)
end

function find_gpu()
    try
        CUDACore.functional() ||
            return nothing, "CUDA is not functional here (CUDACore.versioninfo() says why)"
        dev = CUDACore.device()
        return GPU(CUDACore.name(dev), CUDACore.capability(dev), CUDACore.uuid(dev)), ""
    catch err
        return nothing, sprint(showerror, err)
    end
end

"""
    detect() -> System

Look for everything NeuralBloch needs from the system: TensorRT's runtime
and ONNX parser libraries (loading them to ask their versions), TensorRT's
headers, the CUDA runtime headers they include, g++, and the calling task's
GPU. Never throws; what is missing is `nothing` in the result.
"""
function detect()
    headers = find_headers()
    # The headers' release first, so the library found is the one they match.
    preferred = headers === nothing ? 0 : Int(headers.version.major)
    nvinfer, parser = try
        find_libraries(sort(collect(MAJORS); by = !=(preferred)))
    catch                         # a library without the expected symbols
        nothing, nothing
    end
    gpu, gpu_error = find_gpu()
    return System(nvinfer, parser, headers, find_cuda_headers(), find_compiler(), gpu, gpu_error)
end

function find_libraries(majors)
    h = load_library("libnvinfer", majors)
    h === nothing && return nothing, nothing
    nvinfer = Library(Libdl.dlpath(h), nvinfer_version(h))
    p = load_library("libnvonnxparser", (nvinfer.version.major,))
    return nvinfer, p === nothing ? nothing : Library(Libdl.dlpath(p), parser_version(p))
end

# --- verdict and hints ---------------------------------------------------------

"""
    problems(sys::System; gpu = true) -> Vector{Pair{Symbol,String}}

Everything that keeps NeuralBloch from compiling, building and running
engines here, as `component => explanation`; empty when nothing does. With
`gpu = false` the GPU is not required (what compiling alone needs).
"""
function problems(sys::System; gpu::Bool = true)
    p = Pair{Symbol,String}[]
    lib, hdr, cc = sys.nvinfer, sys.headers, sys.compiler
    lib === nothing && push!(p, :nvinfer => "TensorRT's runtime library (libnvinfer.so.N, N = " *
                                            join(sort(collect(MAJORS)), ", ") * ") not found")
    sys.onnxparser === nothing &&
        push!(p, :onnxparser => "TensorRT's ONNX parser library (libnvonnxparser.so.N) not found")
    if hdr === nothing
        push!(p, :headers => "TensorRT's headers (NvInfer.h, NvOnnxParser.h) not found in " *
                             "/usr/include/$(Sys.ARCH)-linux-gnu or /usr/include")
    elseif lib !== nothing && (hdr.version.major, hdr.version.minor) != (lib.version.major, lib.version.minor)
        push!(p, :headers => "TensorRT's headers in $(hdr.dir) are version $(vstring(hdr.version)), " *
                             "the library is $(vstring(lib.version))")
    end
    if sys.cuda === nothing
        push!(p, :cuda_headers => "the CUDA runtime headers (cuda_runtime_api.h), which TensorRT's " *
                                  "headers include, not found")
    elseif !sys.cuda.complete
        push!(p, :cuda_crt => "$(sys.cuda.dir) has cuda_runtime_api.h but not crt/host_defines.h, " *
                              "which it includes")
    end
    if cc === nothing
        push!(p, :compiler => "g++ not found on PATH")
    elseif cc.version < v"8"
        push!(p, :compiler => "g++ $(cc.version) at $(cc.path) is too old for C++17 (need 8 or newer)")
    end
    gpu && sys.gpu === nothing && push!(p, :gpu => "no functional CUDA GPU: " * sys.gpu_error)
    return p
end

"""
    functional([sys::System]) -> Bool

Whether this machine has everything NeuralBloch needs to compile its C++,
build engines and run them: TensorRT's libraries and headers, the CUDA
runtime headers, g++ and a functional GPU. [`versioninfo`](@ref) shows the
details and how to install what is missing.
"""
functional(sys::System = detect()) = isempty(problems(sys))

const INSTALL_DOCS = "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html"

"""
    hints(components; major = 11, cuda = nothing) -> String

Install commands for the missing `components` (keys of [`problems`](@ref)),
for apt and dnf, from NVIDIA's CUDA repository. `major` is TensorRT's major
version; `cuda` is the CUDA version the TensorRT packages were built for,
which names the CUDA header packages (a placeholder when unknown).
"""
function hints(components; major::Integer = 11, cuda::Union{VersionNumber,Nothing} = nothing)
    xy = cuda === nothing ? "X-Y" : "$(cuda.major)-$(cuda.minor)"
    packages = [  # component => (apt, dnf)
        :nvinfer => (["libnvinfer$major"], ["libnvinfer$major"]),
        :onnxparser => (["libnvonnxparsers$major"], ["libnvonnxparsers$major"]),
        :headers => (["libnvinfer-headers-dev", "libnvonnxparsers-dev"],
                     ["libnvinfer-headers-devel", "libnvonnxparsers-devel"]),
        :cuda_headers => (["cuda-cudart-dev-$xy", "cuda-crt-$xy"], ["cuda-cudart-devel-$xy", "cuda-crt-$xy"]),
        :cuda_crt => (["cuda-crt-$xy"], ["cuda-crt-$xy"]),
        :compiler => (["g++"], ["gcc-c++"]),
    ]
    wanted = [pkgs for (c, pkgs) in packages if c in components]
    lines = String[]
    if !isempty(wanted)
        push!(lines, "Install what is missing from NVIDIA's CUDA repository ($INSTALL_DOCS):",
              "  Debian/Ubuntu: sudo apt-get install " * join(reduce(vcat, first.(wanted)), ' '),
              "  RHEL/Fedora:   sudo dnf install " * join(reduce(vcat, last.(wanted)), ' '))
        if cuda === nothing && (:cuda_headers in components || :cuda_crt in components)
            push!(lines, "  X-Y is the CUDA version your TensorRT packages are built for: X.Y in the " *
                         "+cudaX.Y of their version (apt-cache policy libnvinfer$major, dnf info libnvinfer$major).")
        end
    end
    :gpu in components &&
        push!(lines, "NeuralBloch needs an NVIDIA GPU with a working driver; CUDACore.versioninfo() shows what CUDA sees.")
    return isempty(lines) ? "" : join(lines, '\n') * '\n'
end

"""
    package_cuda(major) -> Union{VersionNumber,Nothing}

The CUDA version the installed `libnvinfer<major>` package was built for,
asked from dpkg or rpm; `nothing` when neither knows the package.
"""
function package_cuda(major::Integer)
    for query in (`dpkg-query -W -f='${Version}' libnvinfer$major`,
                  `rpm -q --qf '%{VERSION}-%{RELEASE}' libnvinfer$major`)
        version = try
            readchomp(pipeline(query; stderr = devnull))
        catch                     # no such package manager, or no such package
            continue
        end
        cuda = parse_package_cuda(version)
        cuda === nothing || return cuda
    end
    return nothing
end

"""
    versioninfo([io::IO], [sys::System])

Print what NeuralBloch found on this machine, what is missing, and the
commands that install it.
"""
function versioninfo(io::IO = stdout, sys::System = detect())
    row(label, detail) = println(io, "  ", rpad(label, 20), something(detail, "not found"))
    lib, parser, hdr, cuda, cc, gpu = sys.nvinfer, sys.onnxparser, sys.headers, sys.cuda, sys.compiler, sys.gpu
    println(io, "NeuralBloch system prerequisites (TensorRT from NVIDIA's deb/rpm packages):")
    row("TensorRT library", lib === nothing ? nothing : "$(vstring(lib.version))  $(lib.path)")
    row("ONNX parser", parser === nothing ? nothing : "$(vstring(parser.version))  $(parser.path)")
    row("TensorRT headers", hdr === nothing ? nothing : "$(vstring(hdr.version))  $(hdr.dir)")
    row("CUDA headers", cuda === nothing ? nothing :
                        cuda.dir * (cuda.complete ? "" : "  (crt/host_defines.h missing)"))
    row("C++ compiler", cc === nothing ? nothing : "g++ $(cc.version)  $(cc.path)")
    row("GPU", gpu === nothing ? "none ($(sys.gpu_error))" :
               "$(gpu.name), compute capability $(gpu.capability.major).$(gpu.capability.minor)")
    probs = problems(sys)
    isempty(probs) && return nothing
    println(io, "\nProblems:")
    foreach(p -> println(io, "  - ", p.second), probs)
    components = first.(probs)
    major = lib !== nothing ? lib.version.major : hdr !== nothing ? hdr.version.major : first(MAJORS)
    cudav = (:cuda_headers in components || :cuda_crt in components) ? package_cuda(major) : nothing
    print(io, "\n", hints(components; major, cuda = cudav))
    return nothing
end
