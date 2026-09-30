# NeuralBloch.jl tests.
#
#     julia --project -e 'using Pkg; Pkg.test()'
#
# One file per module. Each testset belongs to a tier (see setup.jl) and is
# skipped, with the reason printed, when this machine lacks the tier's
# prerequisites; NEURALBLOCH_TEST_ALL=1 turns such skips into failures.
#
# Comparisons against external references and timings are not tests; they
# live in scripts/.

using Test
using NeuralBloch

include("setup.jl")

@testset "NeuralBloch" begin
    include("test_quality.jl")
    include("test_system.jl")
    include("test_cache.jl")
    include("test_compile.jl")
    include("test_build.jl")
    include("test_engine.jl")
    include("test_model.jl")
end
