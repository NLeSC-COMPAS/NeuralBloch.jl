# cache.jl: keys, manifests, locking and atomic publication. Julia only.

using NeuralBloch.TensorRT: digest, engine_key, lookup, getorbuild, withlock,
                            read_manifest, write_manifest

const KEY_FIELDS = (onnx = "3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607",
                    tensorrt = v"11.2.1+2", gpu = "NVIDIA GeForce RTX 2080 Ti", capability = v"7.5",
                    settings = Dict("min" => 1, "opt" => 14400, "max" => 40000, "tf32" => false, "optlevel" => 3),
                    csrc = "c0ffee")

tier(:T0, "cache: keys") do
    k = engine_key(; KEY_FIELDS...)
    @test digest(k) == digest(engine_key(; KEY_FIELDS...))
    @test length(digest(k)) == 64
    # Insertion order does not matter, only content.
    @test digest(Dict(reverse(collect(k)))) == digest(k)

    # Every input changes the key, including every build setting.
    changed = [(; KEY_FIELDS..., onnx = "0" ^ 64), (; KEY_FIELDS..., tensorrt = v"11.2.1+3"),
               (; KEY_FIELDS..., tensorrt = v"11.3.0+1"), (; KEY_FIELDS..., gpu = "NVIDIA A100"),
               (; KEY_FIELDS..., capability = v"8.6"), (; KEY_FIELDS..., csrc = "c0ffef")]
    for s in ("min" => 2, "opt" => 1000, "max" => 65536, "tf32" => true, "optlevel" => 5)
        push!(changed, (; KEY_FIELDS..., settings = merge(KEY_FIELDS.settings, Dict(s))))
    end
    digests = [digest(engine_key(; c...)) for c in changed]
    @test allunique([digests; digest(k)])

    # The key records every input in readable form.
    @test k["tensorrt"] == "11.2.1.2"
    @test k["compute_capability"] == "7.5"
    @test k["settings"]["opt"] == 14400
end

tier(:T0, "cache: manifest") do
    mktempdir() do dir
        key = engine_key(; KEY_FIELDS...)
        info = Dict{String,Any}("seconds" => 4.5, "log" => "line 1\nline \"2\"", "sizes" => [1, 2, 3],
                                "ok" => true)
        write_manifest(dir, key, info)
        m = read_manifest(dir)
        @test m["key"] == key
        @test m["info"] == info
        @test digest(m["key"]) == digest(key)
    end
end

tier(:T0, "cache: build once, publish atomically") do
    mktempdir() do root
        key = engine_key(; KEY_FIELDS...)
        builds = Ref(0)
        build = function (dir)
            builds[] += 1
            write(joinpath(dir, "product"), "engine bytes")
            return Dict{String,Any}("seconds" => 1.0)
        end
        @test lookup(root, key) === nothing
        dir = getorbuild(build, root, key; what = "a test product")
        @test builds[] == 1
        @test read(joinpath(dir, "product"), String) == "engine bytes"
        @test read_manifest(dir)["key"] == key
        @test read_manifest(dir)["info"]["seconds"] == 1.0
        @test lookup(root, key) == dir
        @test getorbuild(build, root, key; what = "a test product") == dir
        @test builds[] == 1
        @test readdir(root) == [basename(dir)]           # no lock, no temporaries left

        # A different key is a different entry.
        other = engine_key(; KEY_FIELDS..., tensorrt = v"11.3.0+1")
        @test getorbuild(build, root, other; what = "a test product") != dir
        @test builds[] == 2
    end
end

tier(:T0, "cache: a failed build leaves nothing") do
    mktempdir() do root
        key = engine_key(; KEY_FIELDS...)
        failing = dir -> (write(joinpath(dir, "half"), "partial"); error("builder exploded"))
        @test_throws "builder exploded" getorbuild(failing, root, key; what = "a test product")
        @test isempty(readdir(root))
        @test lookup(root, key) === nothing
    end
end

tier(:T0, "cache: temporaries are never products") do
    mktempdir() do root
        key = engine_key(; KEY_FIELDS...)
        # What a builder killed mid-way leaves behind: a temporary directory,
        # even one that looks complete.
        final = joinpath(root, digest(key)[1:16])
        stray = final * ".tmp-dead01"
        mkpath(stray)
        write(joinpath(stray, "product"), "stale")
        write_manifest(stray, key, Dict{String,Any}())
        @test lookup(root, key) === nothing
        dir = getorbuild(root, key; what = "a test product") do d
            write(joinpath(d, "product"), "fresh")
            nothing
        end
        @test dir == final
        @test read(joinpath(dir, "product"), String) == "fresh"
        @test !ispath(stray)                              # cleaned up under the lock
    end
end

tier(:T0, "cache: an entry for other fields is replaced") do
    mktempdir() do root
        key = engine_key(; KEY_FIELDS...)
        final = joinpath(root, digest(key)[1:16])
        # Same directory name, different recorded key (a digest-prefix
        # collision, or a hand-edited cache).
        mkpath(final)
        write_manifest(final, engine_key(; KEY_FIELDS..., gpu = "other"), Dict{String,Any}())
        @test lookup(root, key) === nothing
        dir = getorbuild(d -> (write(joinpath(d, "product"), "right"); nothing), root, key; what = "x")
        @test read(joinpath(dir, "product"), String) == "right"
        @test read_manifest(dir)["key"] == key
    end
end

tier(:T0, "cache: waiting for another holder is announced") do
    mktempdir() do root
        lock = joinpath(root, "entry.lock")
        release = Channel{Nothing}(1)
        holder = @async withlock(lock, "the held thing") do
            take!(release)
        end
        while !isfile(lock); sleep(0.01); end
        waiter = @async @test_logs (:info, r"another process is building the held thing"i) match_mode = :any begin
            withlock(() -> :got, lock, "the held thing")
        end
        sleep(0.5)
        @test !istaskdone(waiter)                         # blocked while the lock is held
        put!(release, nothing)
        wait(holder)
        @test fetch(waiter) == :got
        @test !isfile(lock)
    end
end

tier(:T0, "cache: two processes, one build") do
    mktempdir() do root
        counter = joinpath(root, "builds.txt")
        # Each child builds the same entry; the build callback appends to a
        # counter file and takes long enough for the two to overlap.
        code = """
        using NeuralBloch.TensorRT: getorbuild, engine_key
        key = engine_key(; onnx = "x", tensorrt = v"11.2.1+2", gpu = "g", capability = v"7.5",
                         settings = Dict("opt" => 1), csrc = "c")
        dir = getorbuild($(repr(root)), key; what = "the race product") do d
            open(io -> println(io, getpid()), $(repr(counter)), "a")
            sleep(3)
            write(joinpath(d, "product"), string(getpid()))
            nothing
        end
        println(dir)
        print(read(joinpath(dir, "product"), String))
        """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
        procs = [open(pipeline(cmd; stderr = devnull)) for _ in 1:2]
        (dir1, product1), (dir2, product2) = [split(read(p, String), '\n') for p in procs]
        foreach(wait, procs)
        @test all(success, procs)
        @test countlines(counter) == 1                    # the build ran once
        @test dir1 == dir2
        @test product1 == product2 == readchomp(counter)  # both got the one build's product
        @test sort(readdir(root)) == sort([basename(counter), basename(dir1)])
    end
end

tier(:T0, "cache: location") do
    root = NeuralBloch.TensorRT.cache_root()
    @test isdir(root)
    @test startswith(root, first(DEPOT_PATH))
end
