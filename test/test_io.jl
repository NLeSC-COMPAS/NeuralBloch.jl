# io.jl: the I/O contract — engine against card, names, inputs, outputs.
# Julia only: fake engine descriptions, array *types* and plain numbers stand
# in for engines and GPU arrays.

using NeuralBloch: BundleError, publicname, cardname, resolve_io, input_problem, output_problem,
                   layout_problem, overlap_problem, batch_problem, alignment, checkinputs, checkoutputs,
                   outputlayout
using NeuralBloch.TensorRT: TensorInfo
using CUDACore: CuArray, CuMatrix, CuVector, DeviceMemory

x_info(; name = "x", input = true, eltype = Float32, shape = [-1, 3]) =
    TensorInfo(name, input, eltype, shape, input ? ([1, 3], [14400, 3], [40000, 3]) : (Int[], Int[], Int[]))
out_info(name; eltype = Float32, shape = [-1, 600, 2]) = TensorInfo(name, false, eltype, shape, (Int[], Int[], Int[]))
const GOOD = [x_info(), out_info("signal"), out_info("dT1"), out_info("dT2")]   # signal_and_derivatives
const SIGNAL = [x_info(), out_info("signal")]
const COEFFICIENTS = [out_info("coefficients"; shape = [-1, 16]), x_info()]
resolve(tensors, target = :signal_and_derivatives) = resolve_io(tensors, target; ntr = 600, rank = 16)

tier(:T0, "io: names") do
    for (ascii, unicode) in ("T1" => :T₁, "T2" => :T₂, "B1" => :B₁, "signal" => :signal,
                             "dT1" => :∂T₁, "dT2" => :∂T₂, "coefficients" => :coefficients)
        @test publicname(ascii) === unicode
        @test cardname(unicode) == ascii
    end
    @test_throws ArgumentError publicname("T3")
    @test_throws ArgumentError cardname(:T₃)
end

tier(:T0, "io: engine against card") do
    @test resolve(GOOD) == (x = 0, signal = 1, ∂T₁ = 2, ∂T₂ = 3)
    @test resolve(GOOD[[4, 2, 1, 3]]) == (x = 2, signal = 1, ∂T₁ = 3, ∂T₂ = 0)
    @test resolve(SIGNAL, :signal) == (x = 0, signal = 1)
    @test resolve(COEFFICIENTS, :coefficients) == (x = 1, coefficients = 0)
    broken = [  # engine description, target => words the BundleError must contain
        (GOOD[1:3], :signal_and_derivatives) => "no output dT2",
        ([GOOD; out_info("extra")], :signal_and_derivatives) => "unexpected tensor extra",
        ([x_info(input = false); GOOD[2:4]], :signal_and_derivatives) => "no input x",
        ([x_info(eltype = Float16); GOOD[2:4]], :signal_and_derivatives) => "x is Float16",
        ([x_info(shape = [-1, 4]); GOOD[2:4]], :signal_and_derivatives) => "x has shape (-1, 4)",
        ([x_info(shape = [10, 3]); GOOD[2:4]], :signal_and_derivatives) => "x has shape (10, 3)",
        ([GOOD[1]; out_info("signal"; shape = [-1, 599, 2]); GOOD[3:4]], :signal_and_derivatives) =>
            "signal has shape (-1, 599, 2)",
        ([GOOD[1:2]; out_info("dT1"; shape = [-1, 600]); GOOD[4]], :signal_and_derivatives) => "dT1 has shape (-1, 600)",
        ([GOOD[1:3]; out_info("dT2"; eltype = nothing)], :signal_and_derivatives) => "dT2 is not Float32",
        (GOOD, :signal) => "unexpected tensor dT1",                    # the full engine, as the signal's
        (SIGNAL, :signal_and_derivatives) => "no output dT1",
        ([x_info(); out_info("signal"; shape = [-1, 600])], :signal) => "signal has shape (-1, 600)",
        (SIGNAL, :coefficients) => "no output coefficients",
        ([x_info(); out_info("coefficients"; shape = [-1, 15])], :coefficients) => "coefficients has shape (-1, 15)",
        ([x_info(); out_info("coefficients"; shape = [-1, 16, 2])], :coefficients) =>
            "coefficients has shape (-1, 16, 2)",
        ([x_info(); out_info("coefficients"; shape = [-1, 16], eltype = Float16)], :coefficients) =>
            "coefficients is Float16",
        ([COEFFICIENTS; out_info("signal")], :coefficients) => "unexpected tensor signal",
    ]
    expected = (signal_and_derivatives = "outputs signal, dT1 and dT2, Float32 (batch, 600, 2)",
                signal = "output signal, Float32 (batch, 600, 2)",
                coefficients = "output coefficients, Float32 (batch, 16)")
    for ((tensors, target), words) in broken
        err = @test_throws BundleError resolve(tensors, target)
        msg = sprint(showerror, err.value)
        @test occursin(words, msg)
        @test occursin("the $target engine", msg)                        # which engine
        @test occursin("(batch, 3)", msg) && occursin(expected[target], msg)   # what was expected
    end
    @test_throws ArgumentError resolve(GOOD, :derivatives)
end

tier(:T0, "io: input types") do
    @test input_problem(CuVector{Float32}) === nothing
    strided = SubArray{Float32,1,CuArray{Float32,1,DeviceMemory},Tuple{StepRange{Int,Int}},false}
    @test input_problem(strided) === nothing                       # views with a range are fine
    indexed = SubArray{Float32,1,CuArray{Float32,1,DeviceMemory},Tuple{CuArray{Int,1,DeviceMemory}},false}
    @test occursin("strided", input_problem(indexed))
    @test occursin("Float32", input_problem(CuVector{Float64}))
    @test occursin("CPU", input_problem(Vector{Float32}))
    @test occursin("not an array", input_problem(Float32))
end

tier(:T0, "io: output types") do
    for E in (ComplexF32, Float32)
        @test output_problem(CuMatrix{E}, E) === nothing
        @test output_problem(CuArray{E,2,DeviceMemory}, E) === nothing
        @test occursin("CuMatrix{$E}", output_problem(CuVector{E}, E))
        @test occursin("CPU", output_problem(Matrix{E}, E))
        noncontiguous = SubArray{E,2,CuArray{E,2,DeviceMemory},Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}
        @test occursin("contiguous", output_problem(noncontiguous, E))
        @test occursin("not an array", output_problem(Int, E))
    end
    @test occursin("CuMatrix{ComplexF32}", output_problem(CuMatrix{ComplexF64}, ComplexF32))
    @test occursin("CuMatrix{ComplexF32}", output_problem(CuMatrix{Float32}, ComplexF32))     # real for complex
    @test occursin("CuMatrix{Float32}", output_problem(CuMatrix{ComplexF32}, Float32))     # complex for real
end

tier(:T0, "io: batch sizes") do
    @test alignment(600, ComplexF32) == 4               # 4800 bytes a column: 4 columns per 256 bytes
    @test alignment(16, Float32) == 4                   # 64 bytes a column
    @test alignment(32, Float32) == 2
    @test alignment(64, Float32) == 1
    @test alignment(601, ComplexF32) == 32
    b = (min = 10, opt = 64, max = 1000)
    @test batch_problem(b, 0, 4) === nothing                     # nothing to do: fine
    @test batch_problem(b, 10, 4) === nothing
    @test batch_problem(b, 1000, 4) === nothing
    large = batch_problem(b, 1001, 4)
    @test occursin("1001 voxels", large) && occursin("10 to 1000", large)
    @test occursin("Split", large) && occursin("multiples of 4", large) && occursin("max ≥ 1001", large)
    @test occursin("multiples of 2", batch_problem(b, 1001, 2))
    small = batch_problem(b, 5, 4)
    @test occursin("5 voxels", small) && occursin("min ≤ 5", small)
    @test !occursin("Split", small)                              # chunking cannot help a small batch
end

tier(:T0, "io: output layout and overlap") do
    @test layout_problem((600, 100), UInt(0x10000), 600, 100, ComplexF32) === nothing
    @test occursin("size (600, 99)", layout_problem((600, 99), UInt(0x10000), 600, 100, ComplexF32))
    @test occursin("(600, 100)", layout_problem((600, 99), UInt(0x10000), 600, 100, ComplexF32))
    msg = layout_problem((600, 100), UInt(0x10000 + 4800), 600, 100, ComplexF32)   # a view from column 2
    @test occursin("256", msg) && occursin("multiple of 4", msg) && occursin("CuMatrix{ComplexF32}", msg)
    @test layout_problem((600, 100), UInt(0x10000 + 4 * 4800), 600, 100, ComplexF32) === nothing   # column 5

    # Coefficients: 16 Float32 rows, 64 bytes a column
    @test layout_problem((16, 100), UInt(0x10000), 16, 100, Float32) === nothing
    @test occursin("a batch of 100 voxels needs (16, 100)", layout_problem((600, 100), UInt(0x10000), 16, 100, Float32))
    msg = layout_problem((16, 100), UInt(0x10000 + 64), 16, 100, Float32)          # a view from column 2
    @test occursin("multiple of 4", msg) && occursin("CuMatrix{Float32} with 16 rows", msg)
    @test layout_problem((16, 100), UInt(0x10000 + 4 * 64), 16, 100, Float32) === nothing   # column 5

    b = 600 * 100 * 8
    apart = (UInt(0x10000), UInt(0x10000 + b), UInt(0x10000 + 2b))
    @test overlap_problem((:signal, :∂T₁, :∂T₂), apart, b) === nothing
    @test occursin("out.signal and out.∂T₂", overlap_problem((:signal, :∂T₁, :∂T₂),
                   (apart[1], apart[2], apart[1] + 8), b))
    @test occursin("out.∂T₁ and out.∂T₂", overlap_problem((:signal, :∂T₁, :∂T₂),
                   (apart[1], apart[2], apart[2]), b))
end

tier(:T0, "io: inputs and outputs as a whole") do
    p = (T₁ = zeros(Float32, 3), T₂ = zeros(Float32, 3), B₁ = zeros(Float32, 3))
    err = @test_throws ArgumentError checkinputs(p)
    @test occursin("p.T₁ is a Vector{Float32} in CPU memory", sprint(showerror, err.value))
    err = @test_throws ArgumentError checkinputs((T₁ = 1, T₂ = 2))
    @test occursin("p has no property B₁", sprint(showerror, err.value))

    cpu = zeros(ComplexF32, 600, 3)
    check(out) = checkoutputs(out, 3; ntr = 600, rank = 16)
    err = @test_throws ArgumentError check((; signal = cpu))
    @test occursin("out.signal is a Matrix{ComplexF32} in CPU memory", sprint(showerror, err.value))
    err = @test_throws ArgumentError check((; coefficients = zeros(Float32, 16, 3)))
    @test occursin("out.coefficients is a Matrix{Float32} in CPU memory", sprint(showerror, err.value))
    for bad in ((; ∂T₁ = cpu), (; signal = cpu, ∂T₁ = cpu), (; signal = cpu, ∂T₁ = cpu, ∂T₂ = cpu, extra = cpu),
                (; signal = cpu, coefficients = cpu), (; coefficients = cpu, ∂T₁ = cpu, ∂T₂ = cpu))
        err = @test_throws ArgumentError check(bad)
        @test occursin("(; signal, ∂T₁, ∂T₂) in any order, (; signal) or (; coefficients)", sprint(showerror, err.value))
    end
    err = @test_throws ArgumentError check(cpu)                       # not a NamedTuple
    @test occursin("out must be a NamedTuple", sprint(showerror, err.value))
end

tier(:T0, "io: each output set's layout") do
    A = zeros(Float32, 1, 1)
    @test outputlayout((; signal = A), 600, 16) == (600, ComplexF32)
    @test outputlayout((signal = A, ∂T₁ = A, ∂T₂ = A), 600, 16) == (600, ComplexF32)
    @test outputlayout((∂T₂ = A, signal = A, ∂T₁ = A), 600, 16) == (600, ComplexF32)   # any order
    @test outputlayout((; coefficients = A), 600, 16) == (16, Float32)
    @test_throws ArgumentError outputlayout((; signal = A, ∂T₁ = A), 600, 16)
end
