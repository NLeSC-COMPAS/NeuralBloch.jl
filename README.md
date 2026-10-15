# NeuralBloch.jl

NeuralBloch.jl runs the NeuralBloch network on NVIDIA GPUs with TensorRT.
The network is a surrogate for the FISP3D simulation of MR fingerprinting.
From T₁, T₂ and B₁ of each voxel, it predicts the signal and its derivatives
with respect to T₁ and T₂, the signal alone, or the signal's coefficients in
the network's low-rank basis. Each comes from its own TensorRT engine, which
computes only that.

The package contains the trained network as a model bundle in `models/`:
ONNX files and a card that describes them, written by NeuralBloch's
`export_bundle`. The bundled model is `fisp600-rank16` version 1, trained
for T₁ from 0.1 to 6 s, T₂ from 0.01 to 2 s with T₂ < T₁, and B₁ from 0.5
to 1.5. On your machine, the package does these steps automatically:

1. **Compile.** g++ compiles the C++ code of the package (`csrc/`) against
   your TensorRT. This occurs one time for each TensorRT version.
2. **Build.** A subprocess builds a TensorRT engine for your GPU from each
   ONNX file. This occurs one time for each GPU model and engine setting.
3. **Predict.** `predict!` runs the engine for what you ask on your GPU
   arrays. It writes the results directly into output arrays that you own,
   without copies.

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

### Without root access

If you cannot use `sudo`, install TensorRT and the CUDA runtime headers from
conda-forge into a folder that you own, and point the package at that folder
with the environment variable `TENSORRT_ROOT`. The package looks in
`$TENSORRT_ROOT/include`, `$TENSORRT_ROOT/targets/x86_64-linux/include` and
`$TENSORRT_ROOT/lib` before the system directories. You need no
`LD_LIBRARY_PATH`. The compiled code records where the libraries are.

This example uses [pixi](https://pixi.sh). Any conda-compatible tool works
the same way.

1. Choose the CUDA version. Your GPU driver limits it. Run `nvidia-smi` and
   read the CUDA version in the top right (`CUDA Version` or
   `CUDA UMD Version`), for example 13.3. Use exactly that major and minor
   version, as `13.3.*`. With `13.*` or `12.*`, pixi installs the newest
   CUDA of that major, which can be newer than your driver. A TensorRT for
   CUDA 13 does not run on a driver for CUDA 12.x at all.

2. Make a folder with this `pixi.toml`:

   ```toml
   [workspace]
   name = "tensorrt"
   channels = ["conda-forge"]
   platforms = ["linux-64"]

   [dependencies]
   cuda-version = "13.3.*"        # from step 1
   libnvinfer = "11.*"
   libnvinfer-devel = "11.*"
   libnvonnxparser = "11.*"
   libnvonnxparser-devel = "11.*"
   cuda-cudart-dev = "*"
   ```

3. Install the packages, and set `TENSORRT_ROOT` to the environment:

   ```sh
   cd /path/to/that/folder
   pixi install
   export TENSORRT_ROOT=$PWD/.pixi/envs/default
   ```

   Add the `export` line to your `~/.bashrc` so that each new Julia session
   finds TensorRT. Do not start Julia through `pixi run` or `pixi shell`.
   They are not necessary, and they can put conda's compiler on the `PATH`.

4. Start Julia and run `NeuralBloch.versioninfo()`. It must show the
   folder of `TENSORRT_ROOT` for the TensorRT library, the ONNX parser, the
   TensorRT headers and the CUDA headers.

The package uses the system g++ (8 or newer). If your system has no g++,
add `gxx = "*"` to the dependencies and put `$TENSORRT_ROOT/bin` on the
`PATH`.

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
#  the engines. After that, it uses the cache.
engine = Engine()

# Optional: check the model
NeuralBloch.ntr(engine)             # 600 readouts for each voxel
NeuralBloch.rank(engine)            # 16 basis coefficients for each voxel
NeuralBloch.batchprofile(engine)    # (min = 1, opt = 14400, max = 40000) voxels in each call
NeuralBloch.sequence(engine)        # the FISP3D sequence of the training data

# 2. Make the inputs: one GPU vector of Float32 for each parameter.
#    T₁ and T₂ are in seconds. B₁ is relative to nominal.
p = (T₁ = CuArray(Float32[1.0, 0.8, 2.0]),
     T₂ = CuArray(Float32[0.1, 0.05, 0.3]),
     B₁ = CuArray(Float32[1.0, 0.95, 1.1]))

# 3. Predict into new arrays. The target says what to compute.
out = predict(engine, p)                                # (; signal, ∂T₁, ∂T₂), each 600×3
sig = predict(engine, p; target = :signal)              # (; signal) only, much faster
coef = predict(engine, p; target = :coefficients)       # (; coefficients), 16×3
signal = Array(out.signal)                              # Copy to the CPU. This waits for the GPU.

# 4. Or predict into the existing arrays again and again.
out = outputs(engine, 3)                                # or outputs(engine, 3; target = :signal)
for iteration in 1:10
    predict!(out, engine, p)                            # Writes into out.
end

# 5. Release the GPU memory now. If you do not do this, the garbage collector releases it later.
close(engine)
```

Notes:

- The signal and its derivatives are each a `CuMatrix{ComplexF32}`, indexed
  `[readout, voxel]`, at unit proton density. ∂T₁ and ∂T₂ are per second.
  They are not scaled.
- The coefficients are a `CuMatrix{Float32}`, indexed `[coefficient, voxel]`:
  the signal's coordinates in the network's rank-16 basis. The bundle does
  not include the basis.
- `predict!` computes what its outputs ask for: `(; signal, ∂T₁, ∂T₂)`,
  `(; signal)` or `(; coefficients)`.
- The input can be any object with the properties `T₁`, `T₂` and `B₁`.
- For more information about a function, use the Julia help, for example
  `?predict!`.

## Accuracy and speed

Against the training-data generator (the FISP3D EPG simulation of
BlochSimulators.jl, with exact forward-sensitivity derivatives), the
normalized mean absolute error over the training domain is 1.21 % for the
signal, 0.70 % for ∂T₁ and 1.15 % for ∂T₂.

On an RTX 2080 Ti, for 14400 voxels, `predict!` takes 1.28 ms for the signal
and its derivatives, 0.29 ms for the signal alone and 0.13 ms for the
coefficients; for 40000 voxels, 3.5, 0.79 and 0.37 ms. To measure on your
machine, use `scripts/validate.jl`.

## Development

Run the tests:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The tests skip the parts that the machine cannot run, for example the GPU
tests. Set `NEURALBLOCH_TEST_ALL=1` to make each skip a failure.
Update the bundled model with a bundle from NeuralBloch's
`uv run run.py export_bundle`:

```sh
julia --project dev/update_bundle.jl /path/to/NeuralBloch/outputs/bundle
```

It installs the bundle into `models/`. A later version of the same family
must keep the family's contract: the same readouts, rank, sequence and RF
tables, a training domain that is the same or wider, and a higher version.
Otherwise the script changes nothing and lists the differences. Then run the
tests and `scripts/validate.jl`, whose accuracy thresholds belong to the
model they were measured on.
