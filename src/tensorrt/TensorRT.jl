"""
    NeuralBloch.TensorRT

The model-agnostic layer (L1): finds the system TensorRT, compiles the
package's C++ against it, builds and caches engines, and runs them. Nothing
here knows about the NeuralBloch model.
"""
module TensorRT

using CUDACore: CUDACore
using Libdl: Libdl

include("system.jl")

end
