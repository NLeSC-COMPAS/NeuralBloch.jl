# T0: package quality and the public surface. Julia only.

using Aqua
using CUDACore: CUDACore

@testset "Aqua" begin
    # persistent_tasks is broken upstream, not here, on machines with a CUDA
    # driver. There CUDA_Runtime_jll loads the wrapper of the toolkit that
    # suits the driver, which loads CUDA_Compiler_jll, a dependency that
    # CUDA_Runtime_jll ≥ 0.24 omits from its Project.toml (only the registry
    # records it). Since 0.8.18 Aqua builds its precompile subprocess's
    # manifest from each dependency's static Project.toml instead of
    # resolving one, so the subprocess cannot load CUDACore. Without a driver
    # (hosted CI runners) no wrapper loads and the check passes, so it is
    # marked broken only where a wrapper loaded: is_available(). Where it is
    # marked, `broken` keeps it running: the day upstream is fixed it reports
    # an Unexpected Pass, and the marker can go.
    Aqua.test_all(NeuralBloch; persistent_tasks = (broken = CUDACore.CUDA_Runtime_jll.is_available(),))
end

@testset "public names" begin
    # Every name a user may rely on. Adding or removing one is an API change
    # and must update this list.
@test Set(names(NeuralBloch)) == Set([
    :NeuralBloch,
    # exported
    :Engine, :Model, :outputs, :predict!, :predict, :checkdomain,
    # public, qualified
    :ntr, :rank, :domain, :batchprofile, :sequence, :functional, :versioninfo,
    :BundleError, :PrerequisiteError, :CompileError, :BuildError, :TensorRTError])
end
