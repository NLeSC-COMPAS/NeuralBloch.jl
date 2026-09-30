# T0: package quality and the public surface. Julia only.

using Aqua

@testset "Aqua" begin
    Aqua.test_all(NeuralBloch)
end

@testset "public names" begin
    # Every name a user may rely on. Adding or removing one is an API change
    # and must update this list.
    @test Set(names(NeuralBloch)) == Set([:NeuralBloch])
end
