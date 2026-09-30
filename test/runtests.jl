# NeuralBloch.jl tests.
#
#     julia --project -e 'using Pkg; Pkg.test()'
#
# One file per module. The tests are tiered by what the machine provides:
#
#   T0  Julia only                      parsing, cache keys, locking, contract checks, Aqua
#
# Comparisons against external references and timings are not tests; they
# live in scripts/.

using Test
using NeuralBloch

@testset "NeuralBloch" begin
    include("test_quality.jl")
end
