# NeuralBloch.jl development plan

> Status: implemented (2026-10-01). Phase 0 and tasks 1–12 are done, one commit each. `dev/NOTES.md` holds the measurements and the decisions they led to (D1–D19); `README.md` documents the final API.

## Goal

A Julia package that can be installed directly with `Pkg.add`:

- it builds a TensorRT engine on the local machine from a hash-pinned ONNX model;
- it provides a zero-copy `predict!`.

Development and testing do not depend on MRSTAT.jl. Until the switch-over, MRSTAT keeps using its current implementation.

## Verified starting facts (2026-09-30)

### Upstream ONNX

Source: `../NeuralBloch`, branch `tensorRT`, commit `9f4db84`.

- **File**: `outputs/model.onnx`, 1,051,257 bytes.
  - A single file, with no external data.
  - Opset 18, 41 nodes, using only Gemm/Relu/Log/Slice/Concat/Reshape/Shape.
  - sha256 `3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607`.
- **Input `x`**: FP32 (batch, 3), holding the physical values [T₁, T₂, B₁]. The log of T₁/T₂ is taken inside the graph (`src/model.py:42-52`).
- **Outputs `signal`, `dT1`, `dT2`**: FP32 (batch, 600, 2), with (Re, Im) on the last axis (`src/model.py:86-99`).
  - Byte for byte, this is exactly a Julia (600, batch) `ComplexF32` matrix.
- **A ready-made reference**: MRSTAT's `models/model.engine` is byte-identical to upstream's `outputs/model.engine` (sha256 `c5e73aa3…`).
  - It was built from this ONNX, so it can serve as the reference for comparisons.

### Settings of the upstream Python build

See `scripts/build_engine.py`.

- **Only the profile is set explicitly**: MIN/OPT/MAX = 10000/14400/40000 (`config/export.yaml:19-21`, `build_engine.py:56-63`).
  - The comment in `export.yaml` says OPT equals the batch size used in testing (30000). That does not match the actual value, so these numbers are not authoritative.
- **Everything else uses TensorRT 11.2.1.2 defaults**:
  - strongly typed is on;
  - TF32 is on, and never turned off;
  - optimization level 3;
  - workspace equals the whole device memory;
  - no timing cache.

### Upstream has no model card

The information needed is spread over these places:

- **Training domain**
  - T₁ 0.1–5 s and T₂ 0.01–2 s (both log-uniform), B₁ 0.8–1.2 (`config/data.yaml:17-24`).
  - The data keeps only samples with T₁ > T₂ (`data/data_generation.jl:78`). Upstream's verification sampler (`src/utils.py:41-64`) does not apply this filter.
- **Sequence**: FISP3D (`config/data.yaml:5-14`, `data/data_generation.jl:89`).
  - TR 0.00558 s, TE 0.002745 s, TI 0.01084 s, TW 0;
  - max_state 64, 5 repetitions, inversion prepulse, wait spoiling;
  - undersampling factor 1, Δk 0.
  - RF tables: `data/rf_data/rf_angle_list.txt` and `rf_phase_list.txt`, 600 lines each, both in degrees.
- **Derivative convention**: BlochSimulators 0.9 defaults (`data/data_generation.jl:119`).
  - forward difference, ΔT₁ = 0.02 s, ΔT₂ = 0.0025 s;
  - physical units, computed in Float32.

### Upstream has published nothing

No tags, no releases. The repository `NLeSC-COMPAS/NeuralBloch` is private.

### This machine

- **TensorRT 11.2.1.2**, with the full set of deb packages: `libnvinfer11`, `libnvonnxparsers11`, `libnvinfer-headers-dev`, `libnvonnxparsers-dev`.
  - `libnvinfer-bin` is installed too, but this package does not use it.
- **Other software**: CUDA toolkit 13.3 (with cudart-dev), g++ 13.3, Julia 1.12.6, CUDA.jl 6.3.0.
- **GPU**: RTX 2080 Ti (SM 7.5, 11 GB).
- **A compile-time pitfall**: `libnvinfer-headers-dev` depends on nothing, yet `NvInferRuntimeBase.h:24` includes `<cuda_runtime_api.h>`. So compiling also requires `cuda-cudart-dev`.

### Already measured (numbers in appendix A)

- Every CUDA feature `predict!` needs (`CuArray`, `stream`, `device`, `capability`, `reclaim`, `context`, …) is defined in CUDACore.
- Copying the three outputs once costs an extra 55–66% of the engine's own time.
- At batch sizes of 10k–40k, a CUDA graph has the same latency as `enqueueV3`.
  - The graph only saves host submission time (about 130–180 µs down to 10–25 µs), and that time already overlaps with GPU execution.
- Alternating the binding between two sets of output buffers has no measurable cost.

## Three principles

1. **When unsure, measure first.**
   - For every uncertain point, first write a throwaway experiment (Julia or C++) in `dev/spikes/`. Record the numbers and the conclusion in `dev/NOTES.md`, and only then write the real code.
   - Spikes are part of neither the package nor the test suite.
2. **Tests first.**
   - Every task writes its tests before its implementation.
   - One task per commit, in order.
3. **Julia and C++ only.**
   - The package has no Python, shell scripts or CMake, and uses neither PythonCall nor trtexec.
   - All of the following orchestration happens in Julia, replacing MRSTAT's `build_shim.sh`:
     - finding libraries;
     - compiling (via ``run(`g++ …`)``);
     - building the engine;
     - caching and locking.
   - Constraints on the C++ code:
     - only the C++17 standard library and TensorRT's public headers;
     - the exported interface is `extern "C"` and `noexcept`, so C++ exceptions never cross the ABI boundary;
     - TensorRT objects are managed with RAII.
   - System dependencies come only from deb/rpm packages. The package only checks for them and prints installation hints; it never downloads or installs anything.
   - The ONNX is produced upstream. This package only reads that file and never runs upstream code.
   - Tests use Julia only. BlochSimulators appears only in the test environment.

## Interface contract with MRSTAT

NeuralBloch.jl does not depend on MRSTAT, but `predict!` is exactly the interface MRSTAT will call. The points below must be fixed first and written into docstrings. After that, both sides can be developed independently.

| Item | Contract |
|---|---|
| Outputs | Each output is an (NTR, N) `CuMatrix{ComplexF32}`, indexed `[readout, voxel]`, computed at unit proton density. ∂T₁ and ∂T₂ are two separate matrices, in physical units (1/s), not pre-scaled |
| Inputs | Any group of GPU Float32 vectors with properties `T₁`, `T₂`, `B₁`, in physical units. MRSTAT's StructArray can be passed directly |
| Buffers | The caller owns `out`, which may sit at any address. Addresses are rebound on every call: v1 has no graph, and binding has no measurable cost. The `out` arrays must not alias each other. When only the signal is requested, the derivatives go into scratch owned by the Engine |
| Asynchrony | Results are written on the stream of the calling task and become visible in stream order; there is no host synchronization. This is why `checkdomain` is not part of `predict!` |
| Metadata | Public queries for NTR, the batch range and the training domain. MRSTAT's `default_bounds` and `check_experiment` will read these later instead of hard-coding them |
| Batch range | Determined by the profile this package builds with, and the profile is part of the cache key. An out-of-range batch is a hard error, with no padding |

## Phase 0: measure first (spikes)

| # | Question | How to measure | What it decides |
|---|---|---|---|
| S1 | ① Can we compile using only the headers from deb packages (TensorRT's -dev packages, plus `cuda_runtime_api.h` from cudart-dev)? ② With the graph removed, can engine.cpp avoid linking libcudart entirely? ③ Can we `ccall` `getInferLibVersion` and `getNvOnnxParserVersion` directly to get the versions? | Compile a minimal version with g++. Check `nm -u` for undefined `cuda*` symbols and `ldd` for dependencies, then dlopen it in Julia and actually run it once | The prerequisite list and the content of the install hints; whether to link libcudart; how versions are detected (no more `nm`) |
| S2 | When our builder uses the same settings as upstream, do its outputs match MRSTAT's existing engine? | Sample 14400 inputs from the training distribution (T₂ < T₁), run both engines, and compare the maximum relative error | Test tolerances, set from the measured magnitude |
| S3 | Upstream never turned TF32 off; should we? The 2080 Ti itself does not support TF32 | On this machine we can only confirm that on and off give the same results. Once an Ampere or newer GPU is available, compare accuracy and speed with it on and off | The default (recommended: off). The TF32 setting goes into the cache key |
| S4 | With a very wide profile (e.g. 1–65536) instead of 10000–40000, do inference latency, the context's scratch memory or build time change? | Build a few different profiles and measure N = 1000, 14400 and 40000 with the method of appendix A | The default profile; whether `Engine(…; batch = …)` is needed |
| S5 | How long does one build take, and what is its peak GPU memory? Should the workspace be capped? | Time it while watching `nvidia-smi` | The message shown during a build, the subprocess timeout, whether to `CUDA.reclaim()` before building |
| S6 | Can unneeded outputs (dT1/dT2) be bound to a null address, or must they be bound to scratch? | Try it | How the signal-only path is implemented |
| S7 | Does a subprocess build succeed while the parent process holds a CUDA context and memory pool? When the subprocess fails or is killed, is the error clear, and can half a file be left in the cache? Can the GPU index be passed to the subprocess via `CUDA_VISIBLE_DEVICES`? | One experiment per case | Error handling and how to write atomically |
| S8 | When two Julia processes load for the first time at the same moment, is the engine built only once? | Let two processes race for a `FileWatching.Pidfile` lock | Lock parameters and the message shown while waiting |
| S9 | Once S4 fixes the profile, does the graph still bring no benefit at the smallest batch? | Measure with the method of appendix A | Confirms that v1 has no graph |

**Done when**: every item in `dev/NOTES.md` has numbers and a conclusion, and the items that need a decision are clearly listed.

## Package layout

```
NeuralBloch.jl/
├── Project.toml        # CUDACore, Scratch; stdlib: FileWatching, Libdl, SHA, TOML
├── models/             # bundle (see decision 1): model.onnx, card.toml, rf_data/
├── csrc/
│   ├── common.h        # logger, error slot, versions
│   ├── engine.cpp      # no graph: open/close, I/O queries, set shape/address, enqueue
│   └── builder.cpp     # standalone executable: ONNX → .engine
├── src/
│   ├── NeuralBloch.jl  # module, public declarations
│   ├── tensorrt/       # L1 (NeuralBloch.TensorRT): system.jl  cache.jl  compile.jl  build.jl  engine.jl
│   ├── model.jl        # L2: bundle and card
│   ├── io.jl           # L2: contract checks; the only place ASCII ↔ Unicode names are converted
│   └── predict.jl      # L2: Engine, outputs, predict!/predict, checkdomain
├── test/               # one file per module, gated by tier (see test strategy)
├── scripts/validate.jl
└── dev/                # spikes/, NOTES.md
```

## Implementation tasks (one task per commit)

### Phase 1: skeleton and L1 foundations

**1. Repository skeleton**
- What:
  - `Project.toml` with a fresh UUID;
  - the module and its `public` declarations;
  - Aqua. If MRSTAT's known `persistent_tasks` problem shows up, mark it broken the same way;
  - CI runs only T0.
- Tests: the package loads; Aqua passes; the list of public names is as expected.

**2. `system.jl`**
- What:
  - find libnvinfer and libnvonnxparser, and read their versions via `ccall`;
  - check for the headers and g++, and read the GPU name and compute capability;
  - provide `functional()` and `versioninfo()`;
  - print apt/dnf install hints when something is missing.
- Tests:
  - T0: fixed samples for the parsing functions and the hint texts;
  - T1: detection on this machine agrees with what `dpkg` reports.

**3. `cache.jl`**
- What:
  - the cache key consists of the ONNX sha256, the TensorRT version, the GPU name and compute capability, the build settings, and the hash of the csrc sources;
  - the cache lives in a Scratch directory, with a TOML manifest;
  - locking with Pidfile;
  - atomic writes: write a temporary file, then rename it.
- Tests (T0):
  - the same inputs give the same key;
  - changing any one input changes the key;
  - a manifest reads back unchanged after writing;
  - two Julia processes race for the lock, and the test asserts the build callback ran only once;
  - temporary files are never taken for products.

**4. Minimal csrc + `compile.jl`**
- What:
  - compile with g++ into Scratch, cached by source hash;
  - when compilation fails, hand g++'s output to the user verbatim;
  - keep MRSTAT's lesson: no `-static-libstdc++`. When Julia loads the library with `RTLD_DEEPBIND`, a second C++ runtime makes it crash.
- Tests (T1):
  - the exported symbols match the expected list exactly;
  - `ldd` reports no missing libraries;
  - the versions agree with what `system.jl` reads;
  - unchanged sources are not recompiled; changed sources are.

### Phase 2: the build path

**5. Full builder.cpp + `build.jl`**
- What:
  - write out every build setting explicitly: strongly typed, TF32 (per decision 3), optimization level 3, profile;
  - run the builder as a subprocess and collect its stdout/stderr.
- Tests (T2):
  - building from the ONNX in the bundle succeeds;
  - the second call hits the cache and starts no subprocess;
  - changing the profile triggers a rebuild;
  - with a corrupt ONNX, the error includes the parser's messages and no leftovers stay in the cache;
  - with two concurrent processes, the engine is built only once;
  - the manifest records every key field.

### Phase 3: the run path

**6. `engine.cpp` / `engine.jl`**
- What:
  - `TensorRT.Handle`: open/close, with a finalizer;
  - I/O queries;
  - `set_shape`, `set_address`, `run!`.
- Tests (T2):
  - I/O names, shapes and types are correct;
  - a batch outside the profile raises `ArgumentError`;
  - closing twice is safe, and calling after close raises an error instead of crashing;
  - two Handles can coexist;
  - `run!` allocates nothing on the host.

### Phase 4: the model layer

**7. `model.jl`**
- What: read the bundle and verify its sha256. Draft card:
  ```toml
  [model]
  onnx   = "model.onnx"
  sha256 = "3b1645293d530c15f25b8b6b858a3b7fb9ab6e1b8739a40627d8385d516bf607"
  ntr    = 600

  [domain]            # config/data.yaml:17-24; data/data_generation.jl:78
  T1 = [0.1, 5.0]
  T2 = [0.01, 2.0]
  B1 = [0.8, 1.2]
  constraint = "T2 < T1"

  [derivatives]       # BlochSimulators 0.9 defaults; physical units (1/s)
  method   = "forward difference"
  dT1_step = 0.02
  dT2_step = 0.0025

  [sequence]          # config/data.yaml:5-14; data/data_generation.jl:89
  kind = "FISP3D"
  TR = 0.00558
  TE = 0.002745
  TI = 0.01084
  TW = 0.0
  max_state = 64
  repetitions = 5
  inversion_prepulse = true
  wait_spoiling = true
  undersampling_factor = 1
  delta_k_spoil = 0.0
  rf_angle = "rf_data/rf_angle_list.txt"   # degrees
  rf_phase = "rf_data/rf_phase_list.txt"   # degrees

  [provenance]
  upstream = "NeuralBloch@9f4db84"
  blochsimulators = "0.9"
  ```
- Tests (T0):
  - the card parses correctly;
  - missing fields or wrong types give clear errors;
  - a sha256 mismatch raises an error.

**8. `io.jl`**
- What:
  - check that the engine's I/O agrees with the card;
  - convert between ASCII names (T1, dT1, …) and Unicode names (T₁, ∂T₁, …);
  - check `out` for shape, type, contiguity and aliasing.
- Tests (T0): fake I/O descriptions cover every branch of the checks, with no GPU needed.

**9. `predict.jl`**
- What: `Engine`, `outputs`, `predict!`, `predict`, `checkdomain`, and the metadata queries.
- Tests (T2): see the dedicated checklist below.

### Phase 5: validation and release

**10. Accuracy tests + `scripts/validate.jl`**
- Compare against EPG: BlochSimulators 0.9 with the training FISP3D parameters and RF tables.
- The comparison against MRSTAT's existing engine is done only once. Its result goes into `dev/NOTES.md`, not into the test suite.

**11. Distribution test**
- Conditions:
  - a fresh depot (a temporary `JULIA_DEPOT_PATH`);
  - installed with `Pkg.add(url = …)`;
  - a read-only package directory, with no other repositories next to it.
- Done when: `Engine()` compiles and builds automatically, and its predictions are correct.

**12. README**
- prerequisites (apt/dnf commands);
- API;
- what happens on first use;
- performance numbers.

## Test strategy

| Tier | Requires | Tests | Runs where |
|---|---|---|---|
| T0 unit | Julia only | parsing, cache key, locking, contract checks, error messages, Aqua | CI + this machine |
| T1 native | g++ + TensorRT headers and libraries | compilation, exported symbols, linking, versions | this machine |
| T2 GPU | plus a GPU | build, cache, Handle, `predict!` semantics | this machine |
| T3 accuracy | plus BlochSimulators (test environment only) | NMAE against EPG | this machine |
| Performance | — | latency, zero allocations, zero GC | `scripts/validate.jl` (timings are noisy, so not in `Pkg.test`) |

Rules:

- **Skips must say why.** A tier whose prerequisites are missing is skipped automatically, but the reason is printed. With `NEURALBLOCH_TEST_ALL=1`, a skip counts as a failure, so a GPU machine cannot silently miss tests.
- **Tolerances come from measurements.** Every tolerance comes from Phase 0 measurements (for example, a few times the measured error), never from a guess.
- **Every error path has a test.** The test checks both the error type and whether the message contains an actionable hint, such as an install command or the profile range.

Dedicated `predict!` checklist (T2):

- `predict(e, p)` and `predict!(outputs(e, N), e, p)` give bitwise-identical results.
- Repeated calls give bitwise-identical results.
- Alternating between two `out` sets gives bitwise-identical results.
- The signal computed when only the signal is requested is bitwise identical to the signal in the full output.
- When the outputs are column views of a larger matrix (chunked processing), the results equal those of computing the whole block at once.
- `out` with the wrong shape or type, non-contiguous, aliased, or on the CPU: each raises `ArgumentError`, before TensorRT is called.
- On the same stream, a GPU kernel that reads `out` right after `predict!`, with no synchronization, reads correct results.
- After warm-up, `predict!` allocates nothing, neither on the host nor on the GPU.
- `checkdomain`: inputs inside the domain pass; T₁, T₂ or B₁ out of range, or T₂ ≥ T₁, raise `ArgumentError`.

## Public API draft (names are finalized in task 9)

```julia
NeuralBloch.functional(); NeuralBloch.versioninfo()
engine = Engine(Model())               # first use: announce → compile → build (subprocess) → cache
out = outputs(engine, N)               # (; signal, ∂T₁, ∂T₂), each an (NTR, N) CuMatrix{ComplexF32}
out = outputs(engine, N; derivatives = false)   # signal only
predict!(out, engine, p)               # p: GPU Float32 vectors with properties T₁, T₂, B₁
out = predict(engine, p)
checkdomain(engine, p)                 # called explicitly; performs one host synchronization
close(engine)
```

## Decisions needed

| # | Decision | Recommendation | Status |
|---|---|---|---|
| 1 | Where the bundle lives. Upstream has published nothing and the repository is private, so `Artifacts.toml` has no usable download URL for now. Downloading a private release would need a token, so `Pkg` would fail to download it | For v1, put the bundle directly into the package's git repository under `models/`: the ONNX (1.05 MB), plus the card and RF tables. The card records the sha256, which is verified at load time. Switch to Artifacts later, when the bundle grows or several models must be supported | Adopted (`dev/NOTES.md`, D1) |
| 2 | Who writes the card | For v1, write it by hand in this package, with the source of every field (file and line). At the same time, propose a small upstream change: have `export_onnx.py` generate the card during export, and treat the upstream card as authoritative from then on | Adopted for v1 (D2); proposing the upstream change is still open |
| 3 | TF32 | Off by default (accuracy first), and part of the cache key as a build setting. Decide whether to allow turning it on after measuring on an Ampere or newer GPU | Adopted (D3); the measurement on an Ampere or newer GPU is still open |
| 4 | Where the headers come from | Accept only the deb/rpm -dev packages, and drop the fallback of fetching TensorRT headers from GitHub. Reasons: the parser library has to be installed via apt anyway, and the CUDA headers cannot be fetched from GitHub in the first place | Adopted (D4) |
| 5 | Depend on CUDACore or CUDA.jl | CUDACore, already measured to be sufficient. The cost: MRSTAT will later have to tighten its CUDA compat to 6.x, otherwise one environment could end up with two mutually incompatible `CuArray` types | Adopted (D5) |
| 6 | Julia compat | 1.12: the same as MRSTAT, and the only version tested | Adopted (D6) |
| 7 | Where the repository lives, and who can see it | Locally at `../NeuralBloch.jl`. Which GitHub organization, and public or private, is not decided yet; this affects decision 1 | Local only (D7); the GitHub home is still open |

## Out of scope for v1

- CUDA graph: measured to bring no benefit (appendix A). Revisit after re-measuring on a faster GPU.
- Golden I/O: already deferred by an earlier decision. Add it when a check stricter than "compare against EPG" is needed.
- Letting Julia manage TensorRT's scratch memory (the `kUSER_MANAGED` allocation strategy).
- Multiple GPUs, and Windows.
- MRSTAT's adapter and in-place interface: deferred to the switch-over.
- The fallback of fetching headers from GitHub (see decision 4).

## Definition of done

- **Installs and works**: with a fresh depot, a read-only package directory and no other repositories next to it, `Pkg.add` → `Engine()` → `predict!` gives correct results.
- **All tests pass**: T0–T3 all pass on this machine, and T0 passes in CI.
- **`validate.jl` meets its targets**:
  - The NMAE against EPG is within the threshold. The threshold is set from Phase 0 measurements; the reference is MRSTAT's existing engine at 0.56% / 0.25% / 0.45%.
  - `predict!` allocates no memory and triggers no GC.
  - Latency is on par with MRSTAT's existing engine, about 1.3 ms for 14400 voxels.

## Appendix A: measurements of 2026-09-30

Test conditions:

- Hardware and software: RTX 2080 Ti, TensorRT 11.2.1.2.
- Engine: MRSTAT's existing one, NTR = 600, profile 10000/14400/40000.
- Calls: MRSTAT's internal `TensorRTModel` functions (`_stage!`, `_infer!`) and the shim's `set_address` / `enqueue` / `replay`, called directly from the MRSTAT environment.
- Timing: `CUDA.synchronize()` after every call, median of 30–50 calls.
- Scripts:
  - `dev/spikes/bench_predict.jl`: the first table and the GC numbers;
  - `dev/spikes/bench_graph.jl`: the second table.
- How to run: first `source ../MRSTAT.jl/env.sh`, then `julia --project=../MRSTAT.jl dev/spikes/<script>`.
- Re-run: `bench_graph.jl` was re-run from its new location the same day, and the results agreed within noise.

Latency of a single call:

| N | `predict!` (fixed buffers) | `predict`: fresh allocation + rebinding | `predict`: fixed buffers + copy (MRSTAT's current approach) |
|---|---|---|---|
| 10000 | 1.03 ms | 1.23 ms | 1.50 ms |
| 14400 | 1.30 ms | 1.78 ms | 2.12 ms |
| 40000 | 3.43 ms | 3.49 ms | 5.64 ms |

Over 200 consecutive calls:

- `predict!`: 0 GCs.
- Both `predict` variants: 137–549 MiB allocated per call, 3–17 GCs in total, and occasional latency spikes of 1–2.5 ms.

The engine alone (host submission time in parentheses):

| N | graph replay | `enqueueV3`, fixed buffers | `enqueueV3`, alternating two buffer sets | copying 3 outputs |
|---|---|---|---|---|
| 10000 | 1.01 ms (9 µs) | 1.00 ms (137 µs) | 0.91 ms (171 µs) | 0.57 ms |
| 14400 | 1.29 ms (7 µs) | 1.27 ms (127 µs) | 1.27 ms (130 µs) | 0.81 ms |
| 40000 | 3.46 ms (24 µs) | 3.46 ms (181 µs) | 3.48 ms (183 µs) | 2.30 ms |

## Changelog

- 2026-09-30: first draft.
- 2026-09-30: the two benchmark scripts of appendix A were moved into `dev/spikes/`.
- 2026-10-01: implemented. What the implementation changed or added, each with its evidence in `dev/NOTES.md`:
  - the CUDA headers also need `cuda-crt`, which `cuda-cudart-dev` does not pull in (S1);
  - the default profile is 1/14400/40000, and `Engine(model; batch)` builds others (S4);
  - `out` must be 256-byte aligned, as TensorRT documents; this refines "any address" in the contract. Whole arrays always are, and so are column views starting at a multiple of 4 voxels (S6, D13);
  - inputs must be strided GPU vectors, because staging uses strided copies. A spike added after Phase 0 (S10) measured that only strided copies stage without allocating;
  - `checkdomain` checks whichever of T₁, T₂, B₁ `p` has, on GPU or CPU vectors;
  - public metadata queries `ntr`, `domain`, `batchprofile`, `sequence`, and the error types;
  - `predict!` on this machine: bitwise equal to MRSTAT's engine, 1.27–1.30 ms at 14400 voxels, zero allocation (task 10); installs into an empty depot and works there (task 11).
- 2026-10-01: an independent review of the finished code found eleven defects, among them a crash when `close` raced `predict!`. All are fixed, each with a test that failed before the fix (`dev/NOTES.md`, "Review after task 12").
