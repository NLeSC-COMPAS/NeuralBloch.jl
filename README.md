# NeuralBloch.jl

NeuralBloch.jl runs the NeuralBloch network on NVIDIA GPUs with TensorRT.
The network is a surrogate for the FISP3D simulation of MR fingerprinting.
From T₁, T₂ and B₁ of each voxel, it predicts the signal and its derivatives
with respect to T₁ and T₂, the signal alone, or the signal's coefficients in
the network's low-rank basis. Each comes from its own TensorRT engine, which
computes only that.

The trained network comes as a model bundle. The bundle has an ONNX file for
each of these three predictions, the RF tables of the training sequence, and
a card that describes them. The `export_bundle` script of NeuralBloch writes
the bundle, and NeuralBloch publishes it as a release. Pkg downloads the
bundle when it installs the package. The model is `fisp600-rank16` version
0.1.0-beta.1. Its training domain is T₁ from 0.1 to 6 s, T₂ from 0.01 to 2 s
with T₂ < T₁, and B₁ from 0.5 to 1.5. On your machine, the package does these
steps automatically:

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

`Pkg.add` also downloads the model (approximately 1.1 MB) from the
NeuralBloch release
[`model-fisp600-rank16-v0.1.0-beta.1`](https://github.com/NLeSC-COMPAS/NeuralBloch/releases/tag/model-fisp600-rank16-v0.1.0-beta.1),
and checks it against the hashes in `Artifacts.toml`. After that, the package
does not need the network. If your GPU nodes do not have internet access, run
`Pkg.add` (or `Pkg.instantiate`) on a login node that shares the Julia depot.
To use a different model bundle, for example your own export, give its
directory: `Engine(Model("/path/to/bundle"))`.

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
- The coefficients are a `CuMatrix{Float32}`, indexed `[coefficient, voxel]`.
  They are the coordinates of the signal in the rank-16 basis of the network.
  The bundle does not include the basis.
- `predict!` calculates what its outputs ask for: `(; signal, ∂T₁, ∂T₂)` in
  any order, `(; signal)` or `(; coefficients)`.
- The input can be any object with the properties `T₁`, `T₂` and `B₁`.
- For more information about a function, use the Julia help, for example
  `?predict!`.

## Accuracy and speed

The reference is the simulation that made the training data: the FISP3D EPG
simulation of BlochSimulators.jl, with exact forward-sensitivity derivatives.
Against this reference, the normalized mean absolute error over the training
domain is 1.21 % for the signal, 0.70 % for ∂T₁ and 1.15 % for ∂T₂.

On an RTX 2080 Ti, `predict!` takes these times:

| Voxels | Signal and derivatives | Signal | Coefficients |
|---|---|---|---|
| 14400 | 1.28 ms | 0.29 ms | 0.13 ms |
| 40000 | 3.5 ms | 0.79 ms | 0.37 ms |

To measure on your machine, use `scripts/validate.jl`.

## Development

Run the tests:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The tests skip the parts that the machine cannot run, for example the GPU
tests. Set `NEURALBLOCH_TEST_ALL=1` to make each skip a failure.

### Adopt a new model

NeuralBloch publishes each model as a release with the tag
`model-<family>-v<version>`. The release has one file:
`neuralbloch-model-<family>-v<version>.tar.gz`. This package uses only the
model that `Artifacts.toml` binds. A new release does not change this package.

To adopt a new model, do these steps:

1. Download the release file. For example, for `fisp600-rank16` version
   0.1.0-beta.2:

   ```sh
   curl -LO https://github.com/NLeSC-COMPAS/NeuralBloch/releases/download/model-fisp600-rank16-v0.1.0-beta.2/neuralbloch-model-fisp600-rank16-v0.1.0-beta.2.tar.gz
   ```

   If the release is not available, use the file in `outputs/` of
   NeuralBloch. The task `uv run run.py export_bundle` writes this file.

2. Bind the file:

   ```sh
   julia --project dev/bind_bundle.jl neuralbloch-model-fisp600-rank16-v0.1.0-beta.2.tar.gz
   ```

   The script calculates the two hashes of the file. If the depot does not
   have the contents of the file, Pkg installs the file and checks the two
   hashes. Then the script writes the entry of the family in
   `Artifacts.toml`. If the file does not obey the rules in
   [Versions and families](#versions-and-families), the script does not
   change `Artifacts.toml`.

3. Update the values that change with the model:

   | File | Values |
   |---|---|
   | `test/test_model.jl` | Version, ONNX hashes, domain, and `upstream` (the NeuralBloch commit) |
   | `test/test_predict.jl` | Domain |
   | `scripts/validate.jl` | Accuracy thresholds for the new version. First, measure the model with `scripts/validate.jl`. |
   | `test/golden.toml` | Write it again with `julia --project=scripts scripts/golden.jl`. Make sure that the NMAE it gives is near the NMAE of `scripts/validate.jl`. |
   | `README.md` | Version, domain, release link, and "Accuracy and speed" |
   | `dev/NOTES.md` | A record of the model, its file and its measurements |

4. Run the tests with `NEURALBLOCH_TEST_ALL=1`. Then run `scripts/validate.jl`.

5. If the release is not available, publish the file without changes. Refer
   to "Publish a model release" in the README of NeuralBloch. Before you
   upload the file, make sure that its sha256 is the same as the sha256 in
   `Artifacts.toml`.

   CAUTION: Do not run `export_bundle` again after step 2. A new export writes
   a different file.

6. Push the commits after the release is available. Before that time, users
   and the CI cannot install a commit that binds the file.

### Versions and families

A family is a contract for callers. A new version of a family must obey these
rules:

- The readouts, rank, sequence and RF tables do not change.
- The training domain is the same or larger.
- The [semantic version](https://semver.org) is higher, for example
  0.1.0-beta.1, then 0.1.0-beta.2, then 0.1.0.

The parts of the version have these meanings:

- PATCH: the same weights, with a new export.
- MINOR: new weights or a larger domain.
- MAJOR: 0 until the family is stable.

After a change of the card schema MAJOR, `Artifacts.toml` can bind a bundle
that this NeuralBloch does not read. Then `bind_bundle.jl` cannot check the
rules, and refuses the new file. To replace that bundle, add
`--replace-unreadable` before the file name. The script then checks only that
the version is higher.

A model with different readouts, rank or sequence is a new family.
`bind_bundle.jl` adds a new entry for this family, but `Model()` continues to
read `fisp600-rank16`. To change the default family, change the default of
`Model` in `src/model.jl` (`artifact"neuralbloch_<family>"`). Then update the
tests and this README.

CAUTION: Do not remove or replace the file of a published release. Each
commit that binds the file downloads it. To correct a model, publish a new
version.

`bind_bundle.jl` does not bind a second file to a version that
`Artifacts.toml` binds. This rule protects the published versions. To replace
the file of a version that is not published, do these steps:

1. Bind the new file into an empty file:

   ```sh
   rm -f /tmp/Artifacts.toml
   julia --project dev/bind_bundle.jl /path/to/the-new-file.tar.gz /tmp/Artifacts.toml
   ```

2. In `Artifacts.toml`, replace the entry of the family with the entry from
   `/tmp/Artifacts.toml`.
