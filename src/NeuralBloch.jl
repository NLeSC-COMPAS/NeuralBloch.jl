"""
    NeuralBloch

GPU inference for the NeuralBloch surrogate of MR fingerprinting signals: the
package builds a TensorRT engine on this machine from a hash-pinned ONNX model
and predicts signals and their T₁/T₂ derivatives straight into caller-owned
GPU memory.

Two layers:

- `NeuralBloch.TensorRT` (L1) knows nothing about the model: it finds the
  system TensorRT, compiles the package's C++ against it, builds and caches
  engines, and runs them.
- The top level (L2) knows the model: its bundle of ONNX models and card,
  by default the artifact Pkg installs with the package, the contract of its
  inputs and outputs, and `predict!`.
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
