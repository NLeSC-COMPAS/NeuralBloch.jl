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
- The top level (L2) knows the model: the bundled ONNX and its card, the
  contract of its inputs and outputs, and `predict!`.
"""
module NeuralBloch

end
