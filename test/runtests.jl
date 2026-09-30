# NeuralBloch.jl tests.
#
#     julia --project -e 'using Pkg; Pkg.test()'
#
# One file per module. Each testset belongs to a tier (see tiers.jl) and is
# skipped, with the reason printed, when this machine lacks the tier's
# prerequisites; NEURALBLOCH_TEST_ALL=1 turns such skips into failures.
#
# Comparisons against external references and timings are not tests; they
# live in scripts/.

using Test
using NeuralBloch

include("tiers.jl")

@testset "NeuralBloch" begin
    include("test_quality.jl")
    include("test_system.jl")
    include("test_cache.jl")
    include("test_compile.jl")
end
