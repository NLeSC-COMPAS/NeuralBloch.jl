# io.jl: the I/O contract — engine against card, names, inputs, outputs.
# Julia only: fake engine descriptions, array *types* and plain numbers stand
# in for engines and GPU arrays.

using NeuralBloch: BundleError, publicname, cardname, bindings, input_problem, output_problem,
                   layout_problem, overlap_problem, checkinputs, checkoutputs
using NeuralBloch.TensorRT: TensorInfo
using CUDACore: CuArray, CuMatrix, CuVector, DeviceMemory

x_info(; name = "x", input = true, eltype = Float32, shape = [-1, 3]) =
    TensorInfo(name, input, eltype, shape, input ? ([1, 3], [14400, 3], [40000, 3]) : (Int[], Int[], Int[]))
out_info(name; eltype = Float32, shape = [-1, 600, 2]) = TensorInfo(name, false, eltype, shape, (Int[], Int[], Int[]))
const GOOD = [x_info(), out_info("signal"), out_info("dT1"), out_info("dT2")]

tier(:T0, "io: names") do
    for (ascii, unicode) in ("T1" => :T₁, "T2" => :T₂, "B1" => :B₁, "signal" => :signal,
                             "dT1" => :∂T₁, "dT2" => :∂T₂)
        @test publicname(ascii) === unicode
        @test cardname(unicode) == ascii
    end
    @test_throws ArgumentError publicname("T3")
    @test_throws ArgumentError cardname(:T₃)
end

tier(:T0, "io: engine against card") do
    @test bindings(GOOD, 600) == (x = 0, signal = 1, ∂T₁ = 2, ∂T₂ = 3)
    @test bindings(GOOD[[4, 2, 1, 3]], 600) == (x = 2, signal = 1, ∂T₁ = 3, ∂T₂ = 0)
    broken = [  # engine description => words the BundleError must contain
        GOOD[1:3] => "no output dT2",
        [GOOD; out_info("extra")] => "unexpected tensor extra",
        [x_info(input = false); GOOD[2:4]] => "no input x",
        [x_info(eltype = Float16); GOOD[2:4]] => "x is Float16",
        [x_info(shape = [-1, 4]); GOOD[2:4]] => "x has shape (-1, 4)",
        [x_info(shape = [10, 3]); GOOD[2:4]] => "x has shape (10, 3)",
        [GOOD[1]; out_info("signal"; shape = [-1, 599, 2]); GOOD[3:4]] => "signal has shape (-1, 599, 2)",
        [GOOD[1:2]; out_info("dT1"; shape = [-1, 600]); GOOD[4]] => "dT1 has shape (-1, 600)",
        [GOOD[1:3]; out_info("dT2"; eltype = nothing)] => "dT2 is not Float32",
    ]
    for (tensors, words) in broken
        err = @test_throws BundleError bindings(tensors, 600)
        msg = sprint(showerror, err.value)
        @test occursin(words, msg)
        @test occursin("(batch, 3)", msg) && occursin("(batch, 600, 2)", msg)   # what was expected
    end
end

tier(:T0, "io: input types") do
    @test input_problem(CuVector{Float32}) === nothing
    view_type = SubArray{Float32,1,CuArray{Float32,1,DeviceMemory},Tuple{StepRange{Int,Int}},true}
    @test input_problem(view_type) === nothing                     # views are fine: they are copied
    @test occursin("Float32", input_problem(CuVector{Float64}))
    @test occursin("CPU", input_problem(Vector{Float32}))
    @test occursin("not an array", input_problem(Float32))
end

tier(:T0, "io: output types") do
    @test output_problem(CuMatrix{ComplexF32}) === nothing
    @test output_problem(CuArray{ComplexF32,2,DeviceMemory}) === nothing
    @test occursin("CuMatrix{ComplexF32}", output_problem(CuMatrix{ComplexF64}))
    @test occursin("CuMatrix{ComplexF32}", output_problem(CuVector{ComplexF32}))
    @test occursin("CPU", output_problem(Matrix{ComplexF32}))
    noncontiguous = SubArray{ComplexF32,2,CuArray{ComplexF32,2,DeviceMemory},
                             Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}
    @test occursin("contiguous", output_problem(noncontiguous))
    @test occursin("not an array", output_problem(Int))
end

tier(:T0, "io: output layout and overlap") do
    @test layout_problem((600, 100), UInt(0x10000), 600, 100) === nothing
    @test occursin("size (600, 99)", layout_problem((600, 99), UInt(0x10000), 600, 100))
    @test occursin("(600, 100)", layout_problem((600, 99), UInt(0x10000), 600, 100))
    msg = layout_problem((600, 100), UInt(0x10000 + 4800), 600, 100)   # a view from column 2
    @test occursin("256", msg) && occursin("multiple of 4", msg)
    @test layout_problem((600, 100), UInt(0x10000 + 4 * 4800), 600, 100) === nothing   # column 5

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
    err = @test_throws ArgumentError checkoutputs((; signal = cpu), 600, 3)
    @test occursin("out.signal is a Matrix{ComplexF32} in CPU memory", sprint(showerror, err.value))
    for bad in ((; ∂T₁ = cpu), (; signal = cpu, ∂T₁ = cpu), (; signal = cpu, ∂T₁ = cpu, ∂T₂ = cpu, extra = cpu))
        err = @test_throws ArgumentError checkoutputs(bad, 600, 3)
        @test occursin("(; signal) or (; signal, ∂T₁, ∂T₂)", sprint(showerror, err.value))
    end
    @test_throws ArgumentError checkoutputs(cpu, 600, 3)          # not a NamedTuple
end
