# build.jl + builder.cpp: ONNX → engine in a subprocess, cached.

using NeuralBloch.TensorRT: build, BuildSettings, BuildError, PrerequisiteError, run_builder,
                            read_manifest, csrc_hash, vstring
using SHA: sha256

tier(:T0, "build: settings") do
    s = BuildSettings()
    @test (s.batch, s.tf32, s.optlevel) == ((1, 14400, 40000), false, 3)   # dev/NOTES.md D3, D10
    @test BuildSettings(; batch = (1, 64, 1000)).batch == (1, 64, 1000)
    @test TensorRT.settings_dict(s) == Dict("batch" => [1, 14400, 40000], "tf32" => false, "optlevel" => 3)
    for bad in ((0, 1, 2), (10, 5, 20), (1, 50, 20))
        err = @test_throws ArgumentError BuildSettings(; batch = bad)
        @test occursin("1 ≤ min ≤ opt ≤ max", sprint(showerror, err.value))
    end
    @test_throws ArgumentError BuildSettings(; optlevel = 6)
    @test_throws ArgumentError BuildSettings(; optlevel = -1)
end

tier(:T0, "build: needs a GPU") do
    sys = TensorRT.detect()
    nogpu = TensorRT.System(sys.nvinfer, sys.onnxparser, sys.headers, sys.cuda, sys.compiler, nothing, "no driver")
    mktempdir() do root
        err = @test_throws PrerequisiteError build(ONNX; sys = nogpu, root)
        @test occursin("no functional CUDA GPU: no driver", sprint(showerror, err.value))
        @test isempty(readdir(root))
    end
end

tier(:T1, "build: a hung builder is stopped") do
    mktempdir() do dir
        # A stand-in for neuralbloch-build that never finishes.
        write(joinpath(dir, "hang.cpp"), "#include <unistd.h>\nint main() { for (;;) pause(); }\n")
        run(`g++ -o $(joinpath(dir, "hang")) $(joinpath(dir, "hang.cpp"))`)
        native = TensorRT.Native(dir, "", joinpath(dir, "hang"), "")
        gpu = TensorRT.GPU("Some GPU", v"7.5", Base.UUID(0))
        t = @elapsed err = @test_throws BuildError run_builder(native, ONNX, joinpath(dir, "out"), BuildSettings(), gpu;
                                                             timeout = 1)
        @test occursin("did not finish within 1 s", sprint(showerror, err.value))
        @test t < 30

        # Interrupted (Ctrl-C in a notebook reaches only Julia): the builder
        # must not be left running.
        hang = joinpath(dir, "hang")
        task = @async run_builder(native, ONNX, joinpath(dir, "out"), BuildSettings(), gpu; timeout = 600)
        while !success(pipeline(`pgrep -f $hang`; stdout = devnull)); sleep(0.05); end
        schedule(task, InterruptException(); error = true)
        @test_throws TaskFailedException wait(task)
        sleep(0.5)
        @test !success(pipeline(`pgrep -f $hang`; stdout = devnull))
    end
end

tier(:T2, "build: engine from the bundled ONNX, cached") do
    mktempdir() do root
        sys = TensorRT.detect()
        path = @test_logs (:info, r"Building a TensorRT engine") match_mode = :any build(ONNX; root)
        @test isfile(path) && filesize(path) > 100_000

        # The manifest records every key field ...
        m = read_manifest(dirname(path))
        key = m["key"]
        @test sort(collect(keys(key))) ==
              ["compute_capability", "csrc_sha256", "gpu", "onnx_sha256", "settings", "tensorrt"]
        @test key["onnx_sha256"] == bytes2hex(open(sha256, ONNX))
        @test key["tensorrt"] == vstring(sys.nvinfer.version)
        @test key["gpu"] == sys.gpu.name
        @test key["compute_capability"] == "$(sys.gpu.capability.major).$(sys.gpu.capability.minor)"
        @test key["settings"] == Dict("batch" => [1, 14400, 40000], "tf32" => false, "optlevel" => 3)
        @test key["csrc_sha256"] == csrc_hash()
        # ... and what the builder actually applied.
        info = m["info"]
        @test info["batch"] == [1, 14400, 40000]
        @test info["tf32"] == false && info["optlevel"] == 3 && info["strongly_typed"] == true
        @test info["inputs"] == ["x"] && sort(info["outputs"]) == ["dT1", "dT2", "signal"]
        @test info["engine_bytes"] == filesize(path)
        @test info["tensorrt_library"] == vstring(sys.nvinfer.version)

        # The second call hits the cache: no build, no subprocess, same file.
        stamp = mtime(path)
        @test (@test_logs build(ONNX; root)) == path
        @test mtime(path) == stamp

        # Another profile is another engine.
        small = @test_logs (:info, r"Building a TensorRT engine") match_mode = :any build(
            ONNX; settings = BuildSettings(; batch = (1, 64, 1000)), root)
        @test small != path && isfile(small)
        @test read_manifest(dirname(small))["info"]["batch"] == [1, 64, 1000]
    end
end

tier(:T2, "build: a corrupt ONNX") do
    mktempdir() do dir
        root = joinpath(dir, "cache")
        truncated = joinpath(dir, "truncated.onnx")
        write(truncated, read(ONNX)[1:100_000])
        err = @test_throws BuildError build(truncated; root)
        msg = sprint(showerror, err.value)
        @test occursin("the ONNX parser rejected", msg)
        @test occursin("Failed to parse the ONNX model", msg)     # the parser's own words
        @test occursin("truncated.onnx", msg)
        @test isempty(readdir(root))                               # no entry, temporary or lock

        # Text parses "successfully" into an empty network (dev/NOTES.md, S7).
        text = joinpath(dir, "text.onnx")
        write(text, "this is not an ONNX model")
        err = @test_throws BuildError build(text; root)
        @test occursin("no inputs or outputs", sprint(showerror, err.value))
        @test isempty(readdir(root))
    end
end

tier(:T2, "build: two processes, one engine") do
    mktempdir() do dir
        root = joinpath(dir, "cache")
        code = """
        using NeuralBloch.TensorRT: build, BuildSettings
        print(build($(repr(ONNX)); settings = BuildSettings(; batch = (1, 100, 200)), root = $(repr(root))))
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        logs = [joinpath(dir, "stderr$i.txt") for i in 1:2]
        procs = [open(pipeline(cmd; stderr = log)) for log in logs]
        paths = [read(p, String) for p in procs]
        foreach(wait, procs)
        @test all(success, procs)
        @test paths[1] == paths[2] && isfile(paths[1])
        @test count(log -> occursin("Building a TensorRT engine", read(log, String)), logs) == 1
        @test readdir(root) == [basename(dirname(paths[1]))]
    end
end
