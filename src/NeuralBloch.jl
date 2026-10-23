"""
    NeuralBloch

GPU inference for the NeuralBloch surrogate of MR signals. The package builds
TensorRT engines on this machine from hash-pinned ONNX models. The engines
predict the signal, its T₁ and T₂ derivatives, or its basis coefficients
directly into GPU memory that the caller owns.

The package has two layers:

- `NeuralBloch.TensorRT` (L1) does not know the model. It finds the system
  TensorRT, compiles the C++ code of the package against it, builds engines,
  keeps them in a cache, and runs them.
- The top level (L2) knows the model: its bundle of ONNX models and card (by
  default, the artifact that Pkg installs with the package), the contract of
  its inputs and outputs, and `predict!`.
"""
module NeuralBloch

using Artifacts: @artifact_str
using CUDACore: CUDACore, CuArray, CuMatrix, DeviceMemory
using SHA: SHA
using TOML: TOML

include("tensorrt/TensorRT.jl")
include("model.jl")
include("io.jl")
include("predict.jl")

using .TensorRT: functional, PrerequisiteError, CompileError, BuildError, TensorRTError

export Engine, Model, outputs, predict!, predict, checkdomain
public ntr, rank, domain, batchprofile, sequence, functional, versioninfo
public BundleError, PrerequisiteError, CompileError, BuildError, TensorRTError

end
