# NeuralBloch.jl

GPU inference for the NeuralBloch surrogate of MR fingerprinting signals. For
each voxel's T₁, T₂ and B₁, NeuralBloch.jl predicts the FISP3D signal (600
readouts) and its derivatives with respect to T₁ and T₂, with TensorRT,
straight into GPU memory the caller owns.

The package bundles the trained network as a hash-pinned ONNX file and, on
first use, builds a TensorRT engine for your GPU from it; after that it loads
the engine from a cache. There is no Python anywhere: Julia, a little C++
compiled on your machine, and NVIDIA's TensorRT packages.

## Requirements

- Linux on x86_64, an NVIDIA GPU with a working driver, Julia 1.12.
- From NVIDIA's CUDA repository (deb or rpm): TensorRT's runtime libraries
  and headers, the CUDA runtime headers that TensorRT's headers include, and
  g++. On Debian/Ubuntu:

  ```sh
  sudo apt-get install libnvinfer11 libnvonnxparsers11 libnvinfer-headers-dev \
      libnvonnxparsers-dev cuda-cudart-dev-13-3 cuda-crt-13-3 g++
  ```

  On RHEL/Fedora:

  ```sh
  sudo dnf install libnvinfer11 libnvonnxparsers11 libnvinfer-headers-devel \
      libnvonnxparsers-devel cuda-cudart-devel-13-3 cuda-crt-13-3 gcc-c++
  ```

  `13-3` stands for the CUDA version your TensorRT packages are built for: the
  `+cuda13.3` in their version (`apt-cache policy libnvinfer11`). Mind
  `cuda-crt`: `cuda_runtime_api.h` includes a header from it, yet
  `cuda-cudart-dev` does not depend on it.

`NeuralBloch.versioninfo()` checks all of this and prints the commands for
whatever is missing; `NeuralBloch.functional()` returns whether everything is
there.
Nothing is ever installed or downloaded by the package itself.

Tested with TensorRT 11.2.1.2 (`+cuda13.3`), g++ 13.3, CUDACore 6.4 and an
RTX 2080 Ti (compute capability 7.5).

## Installation

The package is not registered. Add it by the URL or local path of its git
repository:

```julia
using Pkg
Pkg.add(url = "/path/to/NeuralBloch.jl")
```

Into an empty depot, installation and the first `Engine()` took four minutes
on the test machine, most of it downloading CUDA artifacts and precompiling.

## Usage

```julia
using NeuralBloch, CUDACore

engine = Engine()                  # the bundled model; compiled and built on first use

# Any object with properties T₁, T₂, B₁: GPU vectors of Float32, in physical
# units (T₁ and T₂ in seconds, B₁ relative to nominal). MRSTAT.jl's
# StructArray of tissue parameters works as it is.
p = (T₁ = CuArray(Float32[1.0, 0.8]), T₂ = CuArray(Float32[0.1, 0.05]), B₁ = CuArray(Float32[1.0, 0.95]))

out = outputs(engine, 2)           # (; signal, ∂T₁, ∂T₂), each a (600, 2) CuMatrix{ComplexF32}
predict!(out, engine, p)           # queued on the task's stream; returns out
out.signal                         # [readout, voxel], unit proton density
out.∂T₁                            # ∂signal/∂T₁, per second (∂T₂ likewise)

predict!(outputs(engine, 2; derivatives = false), engine, p)   # the signal alone
out = predict(engine, p)           # fresh outputs
checkdomain(engine, p)             # throws unless every voxel is in the training domain

NeuralBloch.ntr(engine)            # 600 readouts
NeuralBloch.domain(engine)         # (T₁ = (0.1, 5.0), T₂ = (0.01, 2.0), B₁ = (0.8, 1.2)), and T₂ < T₁
NeuralBloch.batchprofile(engine)   # (min = 1, opt = 14400, max = 40000)
NeuralBloch.sequence(engine)       # the FISP3D sequence of the training data, RF tables included
close(engine)                      # frees its GPU memory now instead of at garbage collection
```

`Engine(model; batch = (min, opt, max), tf32 = false)` builds for another batch
range (the default serves 1 to 40000 voxels per call, tuned for 14400) or
allows TF32 on Ampere and newer GPUs. `Model(dir)` loads another bundle.

### The contract

| | |
|---|---|
| Outputs | Each output is an (NTR, N) `CuMatrix{ComplexF32}` indexed `[readout, voxel]`, at unit proton density. ∂T₁ and ∂T₂ are in physical units (per second), not scaled. |
| Inputs | Properties `T₁`, `T₂`, `B₁` of `p`: strided GPU vectors of Float32 (a `CuVector`, or a view of one with a range) of equal length N, in physical units. |
| Buffers | The caller owns `out`: `(; signal)` or `(; signal, ∂T₁, ∂T₂)`, no two overlapping, each 256-byte aligned as TensorRT requires. Whole arrays are; a column view `A[:, j:k]` is when `j - 1` is a multiple of 4. Addresses are bound on every call, so `out` may change from call to call. Without derivatives, they go to scratch memory the engine owns. |
| Asynchrony | Everything is queued on the calling task's stream, results appear in stream order, and nothing synchronizes the host. That is why `predict!` does not check the domain: `checkdomain` does, and synchronizes. Several tasks may share one engine. |
| Errors | Every contract violation is an `ArgumentError`, thrown before anything is queued. A batch outside `batchprofile(engine)` is one: chunk the batch (column views of `out`), or build an engine for it. |
| Allocation | After warm-up, `predict!` allocates nothing, neither on the host nor on the GPU. |

## What happens on first use

`Engine()` on a new machine, or after TensorRT, the GPU model, a setting or
the package's C++ changed:

1. It checks the system (as `NeuralBloch.versioninfo()` does) and throws a
   `PrerequisiteError` with the install commands if something is missing.
2. It compiles the package's C++ against the installed TensorRT with g++,
   into `libneuralbloch.so` and the `neuralbloch-build` executable (about two
   seconds), announced with one line.
3. It verifies the bundled ONNX against the sha256 its card pins, and builds
   the engine for your GPU in a subprocess (about 5 seconds and 0.5 GiB of GPU
   memory), announced with one line. A failing build is a `BuildError` with
   TensorRT's own messages.
4. Both land in a scratch space of the package in your first Julia depot,
   keyed by everything that determines them: the ONNX, TensorRT's exact
   version, the GPU model, the build settings and the C++ sources. Julia
   processes starting at the same time build once; the others wait, saying so.
   Deleting the cache is always safe.

Later, `Engine()` loads the engine from the cache: about 3 seconds in a fresh
Julia session (CUDA and TensorRT starting up), milliseconds after that. The
engine's execution context holds about 11 KiB of GPU scratch per voxel of the
batch range's maximum: 440 MiB at the default.

## Accuracy

Against the generator of the training data, BlochSimulators.jl 0.9's FISP3D
EPG simulation with finite-difference derivatives, on 14400 voxels drawn from
the training domain, the normalized mean absolute error is 0.56 % for the
signal, 0.25 % for ∂T₁ and 0.45 % for ∂T₂. These are the network's own
errors: on the test machine the engine reproduces MRSTAT.jl's existing
TensorRT engine bit for bit.

## Performance

RTX 2080 Ti, median per call with a device synchronization after each:

| Voxels | `predict!` | engine alone |
|---|---|---|
| 1000 | 0.21–0.25 ms | 0.22 ms |
| 10000 | 0.99 ms | 0.91–0.96 ms |
| 14400 | 1.27–1.30 ms | 1.23 ms |
| 40000 | 3.49 ms | 3.42 ms |

`predict!` stages the inputs and writes the outputs in place: no copies, no
allocation, no garbage collection. `scripts/validate.jl` measures all of this
on your machine.

## The model bundle

`models/` holds the network as trained upstream, in
[NeuralBloch](https://github.com/NLeSC-COMPAS/NeuralBloch) (a private
repository at the time of writing) at commit `9f4db84`: `model.onnx` (input `x`, (batch, 3) `[T₁, T₂, B₁]`; outputs
`signal`, `dT1`, `dT2`, (batch, 600, 2)), the RF tables of the FISP3D
sequence, and `card.toml`, which pins the ONNX's sha256 and records the
training domain, the sequence and the derivative convention, each with its
upstream source.

## Development

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The tests are tiered by what the machine offers: T0 needs Julia only (CI runs
these), T1 adds g++ and TensorRT, T2 a GPU, T3 BlochSimulators (in the test
environment) for accuracy against EPG. A tier the machine cannot run is
skipped with its reason printed; `NEURALBLOCH_TEST_ALL=1` turns skips into
failures.

- `scripts/validate.jl` reports accuracy, allocation and speed against their
  targets. The scripts have an environment of their own:
  `julia --project=scripts -e 'using Pkg; Pkg.instantiate()'` once, then
  `julia --project=scripts scripts/validate.jl`.
- `scripts/distribution.jl` installs the committed package into an empty depot
  and checks that it builds and predicts correctly there.
- `dev/NOTES.md` records the measurements behind every design decision; the
  experiments are in `dev/spikes/`.
- `csrc/` is the C++: `builder.cpp` (ONNX → engine, run as a subprocess),
  `engine.cpp` (the runtime library Julia calls), `common.h`.
