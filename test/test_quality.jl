# T0: package quality and the public surface. Julia only.

using Aqua
using CUDACore: CUDACore

@testset "Aqua" begin
    # On machines with a CUDA driver, persistent_tasks is broken upstream, not
    # here. On these machines, CUDA_Runtime_jll loads the wrapper of the
    # toolkit for the driver. The wrapper loads CUDA_Compiler_jll, but
    # CUDA_Runtime_jll ≥ 0.24 does not list this dependency in its
    # Project.toml (only the registry has it). Since 0.8.18, Aqua makes the
    # manifest of its precompile subprocess from the static Project.toml of
    # each dependency, and does not resolve one. Thus, the subprocess cannot
    # load CUDACore. Without a driver (hosted CI runners), no wrapper loads and
    # the check passes. Thus, the marker applies only where a wrapper loaded:
    # is_available(). With the marker, the check continues to run. When
    # upstream corrects the problem, the check gives an Unexpected Pass. Then
    # remove the marker.
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
