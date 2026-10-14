# system.jl: finding TensorRT, the headers, g++ and the GPU.

tier(:T0, "system: parsing") do
    trt11 = """
    #define NV_INFER_VERSION_H
    #define TRT_MAJOR_ENTERPRISE 11
    #define TRT_MINOR_ENTERPRISE 2
    #define TRT_PATCH_ENTERPRISE 1
    #define TRT_BUILD_ENTERPRISE 2
    #define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE //!< TensorRT major version.
    #define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE //!< TensorRT minor version.
    #define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE //!< TensorRT patch version.
    #define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE //!< TensorRT build number.
    #define NV_TENSORRT_LWS_MAJOR 0 //!< TensorRT LWS major version.
    """
    trt10 = """
    #define NV_TENSORRT_MAJOR 10 //!< TensorRT major version.
    #define NV_TENSORRT_MINOR 3 //!< TensorRT minor version.
    #define NV_TENSORRT_PATCH 0 //!< TensorRT patch version.
    #define NV_TENSORRT_BUILD 26 //!< TensorRT build number.
    """
    @test TensorRT.parse_header_version(trt11) == v"11.2.1+2"
    @test TensorRT.parse_header_version(trt10) == v"10.3.0+26"
    @test TensorRT.parse_header_version("#define NV_TENSORRT_MAJOR 10\n") === nothing
    @test TensorRT.parse_header_version("") === nothing

    withenv("TENSORRT_ROOT" => "/opt/trt") do
        @test TensorRT.root_dirs("include", "lib") == ["/opt/trt/include", "/opt/trt/lib"]
    end
    withenv("TENSORRT_ROOT" => nothing) do
        @test isempty(TensorRT.root_dirs("include"))
    end

    # A TENSORRT_ROOT prefix laid out like a conda environment is searched
    # before the system's directories.
    mktempdir() do root
        include = mkpath(joinpath(root, "include"))
        foreach(f -> write(joinpath(include, f), ""), ("NvInfer.h", "NvOnnxParser.h"))
        write(joinpath(include, "NvInferVersion.h"), trt11)
        cuda = mkpath(joinpath(root, "targets", "$(Sys.ARCH)-linux", "include", "crt"))
        write(joinpath(dirname(cuda), "cuda_runtime_api.h"), "")
        write(joinpath(cuda, "host_defines.h"), "")
        withenv("TENSORRT_ROOT" => root) do
            @test TensorRT.find_headers() == TensorRT.Headers(include, v"11.2.1+2")
            @test TensorRT.find_cuda_headers() == TensorRT.CUDAHeaders(dirname(cuda), true)
        end
    end

    @test TensorRT.parse_compiler_version("13.3.0\n") == v"13.3.0"
    @test TensorRT.parse_compiler_version("12\n") == v"12"
    @test TensorRT.parse_compiler_version("clang version 18") === nothing

    @test TensorRT.parse_package_cuda("11.2.1.2-1+cuda13.3") == v"13.3"     # dpkg
    @test TensorRT.parse_package_cuda("11.2.1.2-1.cuda13.3") == v"13.3"     # rpm
    @test TensorRT.parse_package_cuda("11.2.1.2-1") === nothing

    @test TensorRT.vstring(v"11.2.1+2") == "11.2.1.2"
    @test TensorRT.vstring(v"0.2.0") == "0.2.0"
end

tier(:T0, "system: install hints") do
    all = [:nvinfer, :onnxparser, :headers, :cuda_headers, :compiler]
    h = TensorRT.hints(all; major = 11, cuda = v"13.3")
    @test occursin("sudo apt-get install libnvinfer11 libnvonnxparsers11 libnvinfer-headers-dev " *
                   "libnvonnxparsers-dev cuda-cudart-dev-13-3 cuda-crt-13-3 g++", h)
    @test occursin("sudo dnf install libnvinfer11 libnvonnxparsers11 libnvinfer-headers-devel " *
                   "libnvonnxparsers-devel cuda-cudart-devel-13-3 cuda-crt-13-3 gcc-c++", h)
    @test occursin("docs.nvidia.com/deeplearning/tensorrt", h)
    @test occursin("TENSORRT_ROOT", h)

    # The CUDA version of the TensorRT packages is not always known: then the
    # hint says how to find it.
    h = TensorRT.hints([:cuda_headers]; major = 11, cuda = nothing)
    @test occursin("sudo apt-get install cuda-cudart-dev-X-Y cuda-crt-X-Y", h)
    @test occursin("X.Y", h) && occursin("+cuda", h)

    # cuda-cudart-dev does not depend on cuda-crt (dev/NOTES.md, S1): the
    # hint for a missing crt/host_defines.h names cuda-crt alone.
    h = TensorRT.hints([:cuda_crt]; major = 11, cuda = v"12.9")
    @test occursin("sudo apt-get install cuda-crt-12-9\n", h)
    @test occursin("sudo dnf install cuda-crt-12-9\n", h)

    h = TensorRT.hints([:gpu]; major = 11, cuda = nothing)
    @test occursin("GPU", h) && !occursin("apt-get", h)
    @test TensorRT.hints(Symbol[]; major = 11, cuda = nothing) == ""
end

tier(:T0, "system: report") do
    # A machine with nothing installed: every problem is named, and the
    # report ends with the commands that fix them.
    none = TensorRT.System(nothing, nothing, nothing, nothing, nothing, nothing, "no driver")
    probs = TensorRT.problems(none)
    @test first.(probs) == [:nvinfer, :onnxparser, :headers, :cuda_headers, :compiler, :gpu]
    @test first.(TensorRT.problems(none; gpu = false)) == [:nvinfer, :onnxparser, :headers, :cuda_headers, :compiler]
    @test !TensorRT.functional(none)
    text = sprint(TensorRT.versioninfo, none)
    @test occursin("not found", text)
    @test occursin("sudo apt-get install libnvinfer11 libnvonnxparsers11", text)
    @test occursin("no driver", text)

    # Headers from another TensorRT release than the library.
    lib = TensorRT.Library("/usr/lib/libnvinfer.so.11", v"11.2.1+2")
    parser = TensorRT.Library("/usr/lib/libnvonnxparser.so.11", v"0.2.0")
    cuda = TensorRT.CUDAHeaders("/usr/local/cuda/include", true)
    gxx = TensorRT.Compiler("/usr/bin/g++", v"13.3.0")
    gpu = TensorRT.GPU("Some GPU", v"7.5", Base.UUID(0))
    ok = TensorRT.System(lib, parser, TensorRT.Headers("/usr/include", v"11.2.1+2"), cuda, gxx, gpu, "")
    @test TensorRT.functional(ok)
    @test isempty(TensorRT.problems(ok))
    old = TensorRT.System(lib, parser, TensorRT.Headers("/usr/include", v"10.3.0+26"), cuda, gxx, gpu, "")
    @test first.(TensorRT.problems(old)) == [:headers]
    @test occursin("10.3.0.26", only(TensorRT.problems(old)).second)
    nocrt = TensorRT.System(lib, parser, TensorRT.Headers("/usr/include", v"11.2.1+2"),
                            TensorRT.CUDAHeaders("/usr/include", false), gxx, gpu, "")
    @test first.(TensorRT.problems(nocrt)) == [:cuda_crt]
    ancient = TensorRT.System(lib, parser, TensorRT.Headers("/usr/include", v"11.2.1+2"), cuda,
                              TensorRT.Compiler("/usr/bin/g++", v"6.3.0"), gpu, "")
    @test first.(TensorRT.problems(ancient)) == [:compiler]

    # detect() never throws, whatever the machine.
    @test TensorRT.detect() isa TensorRT.System
    @test NeuralBloch.functional() isa Bool
    @test occursin("TensorRT", sprint(NeuralBloch.versioninfo))
end

# What detection finds must be what the package manager installed.
tier(:T1, "system: detection agrees with dpkg") do
    if Sys.which("dpkg-query") === nothing
        @test_skip "dpkg is not available on this machine"
        return
    end
    if haskey(ENV, "TENSORRT_ROOT")
        @test_skip "TensorRT comes from TENSORRT_ROOT, which no package manager owns"
        return
    end
    sys = TensorRT.detect()
    version(pkg) = readchomp(`dpkg-query -W -f='${Version}' $pkg`)
    owner(path) = first(split(readchomp(`dpkg -S $(realpath(path))`), ':'))
    M = sys.nvinfer.version.major

    @test owner(sys.nvinfer.path) == "libnvinfer$M"
    @test startswith(version("libnvinfer$M"), TensorRT.vstring(sys.nvinfer.version) * "-")
    @test owner(sys.onnxparser.path) == "libnvonnxparsers$M"
    @test owner(joinpath(sys.headers.dir, "NvInfer.h")) == "libnvinfer-headers-dev"
    @test owner(joinpath(sys.headers.dir, "NvOnnxParser.h")) == "libnvonnxparsers-dev"
    @test startswith(version("libnvinfer-headers-dev"), TensorRT.vstring(sys.headers.version) * "-")
    @test startswith(owner(joinpath(sys.cuda.dir, "cuda_runtime_api.h")), "cuda-cudart-dev-")
    @test startswith(owner(joinpath(sys.cuda.dir, "crt", "host_defines.h")), "cuda-crt-")
    gxx = owner(sys.compiler.path)
    @test startswith(version(gxx), string(sys.compiler.version))
    @test TensorRT.package_cuda(M) == TensorRT.parse_package_cuda(version("libnvinfer$M"))
end

tier(:T2, "system: GPU agrees with nvidia-smi") do
    gpu = TensorRT.detect().gpu
    smi = split(readchomp(`nvidia-smi --id=0 --query-gpu=name,compute_cap,uuid --format=csv,noheader`), ", ")
    @test gpu.name == smi[1]
    @test gpu.capability == VersionNumber(smi[2])
    @test "GPU-$(gpu.uuid)" == smi[3]
end
