# compile.jl: compiling csrc/ with g++ into the cache.

using NeuralBloch.TensorRT: compile, csrc_dir, csrc_hash, CompileError, PrerequisiteError
using Libdl: Libdl
using TOML: TOML

# Every symbol libneuralbloch.so exports; everything else is hidden.
const EXPORTS = ["nb_close", "nb_device_memory", "nb_enqueue", "nb_header_version", "nb_io_count",
                 "nb_io_dtype", "nb_io_is_input", "nb_io_name", "nb_io_profile", "nb_io_shape",
                 "nb_last_error", "nb_library_version", "nb_open", "nb_set_address", "nb_set_shape"]

# A private copy of csrc/ to modify.
function csrc_copy(dir)
    src = joinpath(dir, "csrc")
    cp(csrc_dir(), src)
    return src
end

# No cache entry (a directory) was published under `root`.
no_entries(root) = !isdir(root) || !any(f -> isdir(joinpath(root, f)), readdir(root))

tier(:T0, "compile: missing prerequisites") do
    none = TensorRT.System(nothing, nothing, nothing, nothing, nothing, nothing, "no driver")
    mktempdir() do root
        err = @test_throws PrerequisiteError compile(none; root)
        msg = sprint(showerror, err.value)
        @test occursin("TensorRT's runtime library", msg)
        @test occursin("sudo apt-get install libnvinfer11", msg)
        @test !occursin("GPU", msg)                  # compiling needs no GPU
        @test isempty(readdir(root))
    end
end

tier(:T0, "compile: the recipe is part of the key") do
    lib = TensorRT.Library("/usr/lib/libnvinfer.so.11", v"11.2.1+2")
    sys = TensorRT.System(lib, lib, TensorRT.Headers("/usr/include", v"11.2.1+2"),
                          TensorRT.CUDAHeaders("/usr/local/cuda/include", true),
                          TensorRT.Compiler("/usr/bin/g++", v"13.3.0"), nothing, "")
    key = TensorRT.native_key(sys, csrc_dir())
    @test occursin("-O2", key["recipe"]) && occursin("-fvisibility=hidden", key["recipe"])
    @test sort(collect(keys(key))) == ["arch", "compiler", "csrc_sha256", "recipe", "tensorrt", "tensorrt_headers"]
end

tier(:T0, "compile: source hash") do
    mktempdir() do dir
        src = csrc_copy(dir)
        h = csrc_hash(src)
        @test h == csrc_hash(csrc_dir())
        touch(joinpath(src, "engine.cpp"))            # content, not timestamps
        @test csrc_hash(src) == h
        write(joinpath(src, "engine.cpp"), read(joinpath(src, "engine.cpp"), String) * "// edit\n")
        @test csrc_hash(src) != h
        h2 = csrc_hash(src)
        write(joinpath(src, "extra.h"), "")
        @test csrc_hash(src) != h2
    end
end

tier(:T1, "compile: products") do
    mktempdir() do root
        native = @test_logs (:info, r"Compiling") match_mode = :any compile(; root)
        sys = TensorRT.detect()

        # Exactly the extern "C" API is exported.
        nm = readchomp(`nm -D --defined-only $(native.library)`)
        @test sort([last(split(l)) for l in split(nm, '\n')]) == EXPORTS

        # Everything it links resolves, and nothing links libcudart
        # (dev/NOTES.md, S1): TensorRT carries its own CUDA runtime.
        for f in (native.library, native.builder)
            deps = read(`ldd $f`, String)
            @test !occursin("not found", deps)
            @test !occursin("libcudart", deps)
            @test occursin("libnvinfer.so.$(sys.nvinfer.version.major)", deps)
        end
        @test occursin("libnvonnxparser.so", read(`ldd $(native.builder)`, String))

        # The versions both products report are the ones system.jl found.
        lib = Libdl.dlopen(native.library)
        version(f) = (v = zeros(Int32, 4); ccall(Libdl.dlsym(lib, f), Cvoid, (Ptr{Int32},), v);
                      VersionNumber(v[1], v[2], v[3], (), (v[4],)))
        @test version(:nb_header_version) == sys.headers.version
        @test version(:nb_library_version) == sys.nvinfer.version
        @test unsafe_string(ccall(Libdl.dlsym(lib, :nb_last_error), Cstring, ())) == ""
        report = TOML.parse(read(`$(native.builder) --version`, String))
        @test report["tensorrt_header"] == TensorRT.vstring(sys.headers.version)
        @test report["tensorrt_library"] == TensorRT.vstring(sys.nvinfer.version)
        @test report["onnx_parser"] == TensorRT.vstring(sys.onnxparser.version)

        # The manifest says how the products were made.
        info = TensorRT.read_manifest(native.dir)["info"]
        @test occursin("g++", info["library_command"]) && occursin("g++", info["builder_command"])
        @test !occursin("static-libstdc++", info["library_command"])
    end
end

tier(:T1, "compile: cached by source") do
    mktempdir() do dir
        src, root = csrc_copy(dir), joinpath(dir, "cache")
        a = compile(; csrc = src, root)
        stamp = mtime(a.library)
        b = @test_logs compile(; csrc = src, root)    # no "Compiling": nothing is compiled
        @test b.dir == a.dir && mtime(b.library) == stamp

        write(joinpath(src, "engine.cpp"), read(joinpath(src, "engine.cpp"), String) * "// edit\n")
        c = @test_logs (:info, r"Compiling") match_mode = :any compile(; csrc = src, root)
        @test c.dir != a.dir
        @test isfile(c.library) && isfile(c.builder)
    end
end

tier(:T1, "compile: the headers g++ uses must be TensorRT's") do
    # As if another TensorRT's headers came first on g++'s include path
    # (CPATH, /usr/local/include): what g++ compiled against is not what
    # detect() found. Same major.minor, so detection itself has no objection.
    sys = TensorRT.detect()
    v = sys.headers.version
    other = TensorRT.System(sys.nvinfer, sys.onnxparser,
                            TensorRT.Headers(sys.headers.dir, VersionNumber(v.major, v.minor, v.patch + 1, (), v.build)),
                            sys.cuda, sys.compiler, sys.gpu, sys.gpu_error)
    mktempdir() do root
        err = @test_throws PrerequisiteError compile(other; root)
        msg = sprint(showerror, err.value)
        @test occursin(TensorRT.vstring(v), msg) && occursin("CPATH", msg)
        @test no_entries(root)
    end
end

tier(:T1, "compile: g++ errors reach the user verbatim") do
    mktempdir() do dir
        src, root = csrc_copy(dir), joinpath(dir, "cache")
        write(joinpath(src, "engine.cpp"),
              read(joinpath(src, "engine.cpp"), String) * "\nint broken = this_symbol_does_not_exist;\n")
        err = @test_throws CompileError compile(; csrc = src, root)
        msg = sprint(showerror, err.value)
        @test occursin("this_symbol_does_not_exist", msg) && occursin("was not declared", msg)  # g++'s words
        @test occursin("engine.cpp", msg)
        @test occursin("TensorRT", msg)                  # what to suspect, beyond g++'s words
        @test no_entries(root)
    end
end
