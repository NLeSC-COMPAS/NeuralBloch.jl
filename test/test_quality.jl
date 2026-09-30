# T0: package quality and the public surface. Julia only.

using Aqua

@testset "Aqua" begin
    # persistent_tasks is broken upstream, not here (the same marker as
    # MRSTAT.jl's): since 0.8.18 Aqua builds its precompile subprocess's
    # manifest from each dependency's static Project.toml instead of
    # resolving one, and CUDA_Runtime_jll ≥ 0.24 omits CUDA_Compiler_jll
    # there (only the registry records it), so the subprocess cannot load
    # CUDACore. `broken` keeps the check running: the day upstream is fixed it
    # reports an Unexpected Pass — drop the marker then.
    Aqua.test_all(NeuralBloch; persistent_tasks = (broken = true,))
end

@testset "public names" begin
    # Every name a user may rely on. Adding or removing one is an API change
    # and must update this list.
@test Set(names(NeuralBloch)) == Set([
    :NeuralBloch,
    # exported
    :Engine, :Model, :outputs, :predict!, :predict, :checkdomain,
    # public, qualified
    :ntr, :domain, :batchprofile, :sequence, :functional, :versioninfo,
    :BundleError, :PrerequisiteError, :CompileError, :BuildError, :TensorRTError])
end
