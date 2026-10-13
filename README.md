# NeuralBloch.jl

NeuralBloch.jl runs the NeuralBloch network on NVIDIA GPUs with TensorRT.
The network is a surrogate for the FISP3D simulation of MR fingerprinting.
From T₁, T₂ and B₁ of each voxel, it predicts the signal and its derivatives
with respect to T₁ and T₂.

The package contains the trained network as an ONNX file. On your machine,
the package does these steps automatically:

1. **Compile.** g++ compiles the C++ code of the package (`csrc/`) against
   your TensorRT. This occurs one time for each TensorRT version.
2. **Build.** A subprocess builds a TensorRT engine for your GPU from the
   ONNX file. This occurs one time for each GPU model and engine setting.
3. **Predict.** `predict!` runs the engine on your GPU arrays. It writes the
   results directly into output arrays that you own, without copies.

A cache keeps the results of steps 1 and 2. You can delete the cache at any
time. For more information, see [`architecture.md`](architecture.md).

## Requirements

- Linux on x86_64, an NVIDIA GPU with a working driver, and Julia 1.12.
- TensorRT, the CUDA runtime headers and g++, from the NVIDIA CUDA
  repository. On Debian or Ubuntu, run:

  ```sh
  sudo apt-get install libnvinfer11 libnvonnxparsers11 libnvinfer-headers-dev \
      libnvonnxparsers-dev cuda-cudart-dev-13-3 cuda-crt-13-3 g++
  ```

  On RHEL or Fedora, use `dnf` and the `-devel` names, and `gcc-c++`.
  Replace `13-3` with the CUDA version of your TensorRT packages (look for
  `+cuda13.3` in `apt-cache policy libnvinfer11`).

The package does not install these packages for you. To examine your system,
run `NeuralBloch.versioninfo()`. It shows the install command for each
missing package.

## Installation

The package is not in the Julia General registry. Add it with the URL or
the local path of its git repository. You also need CUDACore.jl for GPU
arrays.

```julia
using Pkg
Pkg.add(url = "/path/to/NeuralBloch.jl")
Pkg.add("CUDACore")
```

## Quick start

This example shows the complete workflow. Run it as one script.

```julia
using NeuralBloch, CUDACore

# 1. Make the engine. The first time, this compiles the C++ code and builds
#  the engine. After that, it uses the cache.
engine = Engine()

# Optional: check the model
NeuralBloch.ntr(engine)             # 600 readouts for each voxel
NeuralBloch.batchprofile(engine)    # (min = 1, opt = 14400, max = 40000) voxels in each call
NeuralBloch.sequence(engine)        # the FISP3D sequence of the training data

# 2. Make the inputs: one GPU vector of Float32 for each parameter.
#    T₁ and T₂ are in seconds. B₁ is relative to nominal.
p = (T₁ = CuArray(Float32[1.0, 0.8, 2.0]),
     T₂ = CuArray(Float32[0.1, 0.05, 0.3]),
     B₁ = CuArray(Float32[1.0, 0.95, 1.1]))

# 3. Predict into new arrays.
out = predict(engine, p)                         # (; signal, ∂T₁, ∂T₂), each 600×3
sig = predict(engine, p; derivatives = false)    # (; signal) only
signal = Array(out.signal)                       # Copy to the CPU. This waits for the GPU.

# 4. Or predict into the existing arrays again and again.
out = outputs(engine, 3)                         # or outputs(engine, 3; derivatives = false)
for iteration in 1:10
    predict!(out, engine, p)                     # Writes into out.
end

# 5. Release the GPU memory now. If you do not do this, the garbage collector releases it later.
close(engine)
```

Notes:

- Each output is a `CuMatrix{ComplexF32}`, indexed `[readout, voxel]`, at
  unit proton density. ∂T₁ and ∂T₂ are per second. They are not scaled.
- The input can be any object with the properties `T₁`, `T₂` and `B₁`.
- For more information about a function, use the Julia help, for example
  `?predict!`.

## Accuracy and speed

Against the training-data generator (the FISP3D EPG simulation of
BlochSimulators.jl), the normalized mean absolute error is 0.56 % for the
signal, 0.25 % for ∂T₁ and 0.45 % for ∂T₂.

On an RTX 2080 Ti, `predict!` takes 1.3 ms for 14400 voxels and 3.5 ms for
40000 voxels. To measure on your machine, use `scripts/validate.jl`.

## Development

Run the tests:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The tests skip the parts that the machine cannot run, for example the GPU
tests. Set `NEURALBLOCH_TEST_ALL=1` to make each skip a failure.