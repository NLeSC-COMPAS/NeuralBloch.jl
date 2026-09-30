# Phase 0 notes

Measured 2026-09-30 on this machine: RTX 2080 Ti (SM 7.5, 11 GiB, driver
610.57.04), TensorRT 11.2.1.2 from NVIDIA's deb packages, CUDA toolkit 13.3,
g++ 13.3, Julia 1.12.6, CUDACore 6.4.0. The ONNX is upstream's
`outputs/model.onnx` (NeuralBloch `tensorRT` branch, commit `9f4db84`,
sha256 `3b164529…`); the reference engine is MRSTAT's `models/model.engine`
(sha256 `c5e73aa3…`, byte-identical to upstream's).

Every spike runs from the repository root in the `dev/` environment
(CUDACore only) and writes its products to `.work/spikes/` (gitignored):

    julia --project=dev dev/spikes/<spike>.jl

`spike_engine.cpp` and `spike_builder.cpp` are throwaway C++ helpers that
`common.jl` compiles with g++. None of this is package code.

Latency is measured as in appendix A of `plan.md`: a device synchronize after
every call, median of 50 calls; "host" is the submission alone, queue drained.

## S1 — headers, linking, versions (`s1_link.jl`)

**① Headers.** The deb packages suffice. TensorRT's headers
(`libnvinfer-headers-dev`, `libnvonnxparsers-dev`) live in
`/usr/include/x86_64-linux-gnu`, which is on g++'s default search path, so
they need no `-I`. `NvInferRuntimeBase.h:24` includes `<cuda_runtime_api.h>`,
which needs `-isystem /usr/local/cuda/include` and pulls in nine CUDA headers:

| Header | Package |
|---|---|
| `cuda_runtime_api.h`, `builtin_types.h`, `driver_types.h`, `vector_types.h`, … (8) | `cuda-cudart-dev-13-3` |
| `crt/host_defines.h` | **`cuda-crt-13-3`** |

`cuda-crt-13-3` is **not** a dependency of `cuda-cudart-dev-13-3` (only
`cuda-nvcc`/`cuda-compiler` pull it in), so a machine without nvcc fails to
compile with only cudart-dev installed. The plan's "compile-time pitfall"
has a second half.

**② Linking.** Neither a graph-free runtime library nor the builder has any
undefined `cuda*` symbol (`nm -u`). `ldd` shows only `libnvinfer.so.11`
(plus `libnvonnxparser.so.11` for the builder) and the C/C++ runtime.
TensorRT links its CUDA runtime statically. **Do not link libcudart.**

**③ Versions.** Both can be read with a plain `ccall` into the system
libraries, so `nm` is no longer needed:

| Call | Result | Meaning |
|---|---|---|
| `getInferLibVersion()` (`libnvinfer.so.11`) | `110201` | 11.2.1 (`major·10⁴ + minor·10² + patch`); `getInferLibBuildVersion()` gives the 4th field (2) |
| `getNvOnnxParserVersion()` (`libnvonnxparser.so.11`) | `200` | **0.2.0**: in TensorRT 11 the ONNX parser has its own version scheme |

The parser's version cannot be compared with TensorRT's; the two are paired
by their soname major (`.so.11`), as the deb packages pin them to each other.

**Conclusions**
- Prerequisites (Debian/Ubuntu): `libnvinfer11 libnvonnxparsers11` (runtime),
  `libnvinfer-headers-dev libnvonnxparsers-dev` (compile),
  `cuda-cudart-dev-13-3 cuda-crt-13-3` (the CUDA headers TensorRT's headers
  include), `g++`. `libnvonnxparsers-dev` alone pulls in all the TensorRT
  ones through `libnvinfer-dev`.
- Link with `-lnvinfer` (engine library) and `-lnvinfer -lnvonnxparser`
  (builder) only; no CUDA library, no rpath into the CUDA toolkit.
- Versions come from `ccall`: 4 fields from the four `getInferLib*Version`
  functions, the parser's own version for display only.

## S2 — our builder vs MRSTAT's engine (`s2_parity.jl`)

Our builder with upstream's settings (profile 10000/14400/40000, everything
else at TensorRT's defaults), built twice, against MRSTAT's engine, on
identical inputs from the training distribution (T₂ < T₁, fixed seeds):

| N | signal | dT1 | dT2 |
|---|---|---|---|
| 10000 | bitwise equal (both builds) | bitwise equal | bitwise equal |
| 14400 | bitwise equal | bitwise equal | bitwise equal |
| 40000 | bitwise equal | bitwise equal | bitwise equal |

- The engine files differ (1159876 vs 1159868 bytes), and so do our two
  builds (627 bytes): tactic selection is timing-based. 9 of the 16 GEMM
  kernel choices differ from MRSTAT's (`turing_sgemm_128x64` vs `64x64`);
  these tilings accumulate in the same order, hence identical results.
- Repeated calls on one engine are bitwise identical.
- Scratch memory: 439 MiB for all three engines.

**Conclusions**
- The builder reproduces upstream's engine. No numerical tolerance is needed
  to compare against it at these settings.
- Engines are not reproducible byte for byte, so no test compares engine
  files; the cache key describes inputs, never output bytes.
- Test tolerances: bitwise equality within one engine (repeated calls,
  alternating buffers, chunks — see S6 ③b); between different builds, the
  largest difference measured anywhere in Phase 0 is 1.0e-6 relative (S4,
  OPT 1000 vs 14400), so cross-build comparisons use **5e-6**.

## S3 — TF32 (`s3_tf32.jl`)

TensorRT enables TF32 by default (`default_tf32 = 1`). On the 2080 Ti
(SM 7.5, no TF32 hardware) an engine built with TF32 off gives bitwise
identical outputs at N = 10000/14400/40000, no kernel names mention tf32,
and latency is the same within noise (14400: 1.26 vs 1.23 ms).

**Conclusions**
- On this machine TF32 is a no-op, so turning it off costs nothing here.
- Default **off** (accuracy first; decision 3) and part of the cache key.
- Still open: accuracy and speed with TF32 on, on an Ampere-or-newer GPU.

## S4 — profile width (`s4_profiles.jl`)

| MIN/OPT/MAX | build s | scratch MiB | N=1000 | N=14400 | N=40000 | N=65536 | outputs vs 10000/14400/40000 |
|---|---|---|---|---|---|---|---|
| 10000/14400/40000 | 3.46 | 439 | — | 1.347 | 3.400 | — | (reference) |
| 1/14400/40000 | 3.44 | 439 | 0.226 | 1.380 | 3.579 | — | bitwise |
| 10000/14400/65536 | 3.50 | 720 | — | 1.412 | 3.517 | 5.586 | bitwise |
| 1/14400/65536 | 3.51 | 720 | 0.236 | 1.424 | 3.472 | 5.561 | bitwise |
| 1/1000/65536 | 3.27 | 448 | 0.203 | 1.765 | 4.110 | 6.624 | ≤ 1.0e-6 rel |

Latencies in ms. The candidates again, interleaved round by round (median of
7 rounds × 50 calls, range in parentheses):

| N | 10000/14400/40000 | 1/14400/40000 | 1/14400/65536 |
|---|---|---|---|
| 14400 | 1.235 (1.234–1.309) | 1.236 (1.234–1.304) | 1.270 (1.268–1.279) |
| 40000 | 3.349 (3.337–3.407) | 3.422 (3.353–3.428) | 3.484 (3.410–3.491) |

- Build time does not depend on the profile (3.3–3.5 s).
- Scratch memory follows MAX (≈ 11 KiB per voxel of MAX) and the tactics.
- MIN = 1 costs nothing measurable; the ≤ 2% differences are the size of
  the build-to-build tactic noise (the two OPT-14400/MAX-40000 engines picked
  different GEMM kernels).
- MAX 65536 costs +281 MiB scratch and ≈ 3–4% latency.
- OPT matters: OPT 1000 makes N = 14400 31% slower (1.765 vs 1.347 ms in the
  same table) and changes results by up to 1.0e-6 (different GEMM tiling).

**Conclusions**
- Default profile **MIN/OPT/MAX = 1/14400/40000**: every batch up to 40000
  (small test problems, chunk tails) at upstream's speed and memory.
- `Engine(…; batch = …)` is needed: MAX is the one real trade-off (memory
  against range), and it goes into the cache key.

## S5 — build cost, workspace (`s5_build_cost.jl`)

Profile 1/14400/40000, 3 builds per setting, the builder as the only GPU user:

| Workspace limit | wall s | build s | peak GPU MiB | peak host RSS MiB |
|---|---|---|---|---|
| default (= device memory, 10.6 GiB) | 4.73 | 3.49 | 474 | 1360 |
| 1024 MiB | 4.67 | 3.48 | 474 | 1360 |
| 512 MiB | 4.69 | 3.52 | 474 | 1359 |
| 448 MiB | 4.68 | 3.50 | 474 | 1360 |
| 440 MiB | 4.64 | 3.49 | 474 | 1360 |
| 400 MiB | build fails | | | |
| 256 MiB | build fails | | | |

The failure: `Error Code 10: Internal Error (Could not find any implementation
for node {ForeignNode[subnet_signal.net.0.weight...node_view_1]})`, preceded by
`Skipping tactic 0 due to insufficient memory`. The fused node's scratch
counts against the workspace limit. Every engine that built has 439 MiB
scratch, and all give bitwise identical outputs.

**Conclusions**
- A build takes ≈ 5 s wall: announce it with one `@info` line; no progress
  display. The subprocess timeout only guards against hangs (10 minutes).
- Do not cap the workspace: below the engine's scratch the build fails,
  above it nothing changes. Keep TensorRT's default.
- The build needs ≈ 0.5 GiB of free device memory at MAX 40000; see S7 for
  `reclaim`.

## S6 — outputs: null, shared, misaligned (`s6_outputs.jl`)

**① Null addresses.** `setTensorAddress(dT1, nullptr)` is accepted (it resets
the binding), but `enqueueV3` then fails: `Neither address or allocator is
set for output tensor dT1`. Unwanted outputs need real memory.

**② One shared scratch.** dT1 and dT2 bound to the *same* buffer: enqueue
succeeds and the signal is bitwise equal to a full run, 20 times in a row.

**③ Alignment.** The header (`setTensorAddress`) says: "The pointer must have
at least 256-byte alignment." A column view `A[:, j:k]` of an (600, M)
ComplexF32 matrix starts `(j-1)·4800` bytes in, which is 256-aligned only
when `j-1` is a multiple of 4. TensorRT 11.2 on SM 7.5 accepts offsets with
remainders 64/128/192 and computes bitwise correct results anyway, i.e. the
requirement is documented but not enforced here. Fresh `CuArray`s are
256-aligned.

**③b Chunks.** The same 14400 voxels computed in chunks (7200+7200;
4×~3600; 5001+9399; 1+1+1+97+9899+4401) into column views of one matrix are
bitwise equal to the whole block: each voxel's result does not depend on the
batch it is computed in.

**Conclusions**
- Signal-only calls bind dT1 and dT2 to **one** scratch buffer owned by the
  Engine (grow-only, reused through leading-column views, which keep its
  alignment).
- `predict!` checks the 256-byte alignment of every `out` array and throws
  `ArgumentError` for a misaligned one, with the rule in the message: the
  measured tolerance is undocumented behaviour, so the package does not rely
  on it. For NTR = 600, chunk boundaries at multiples of 4 voxels are aligned.
- The chunked-processing test can require bitwise equality.

## S7 — the builder as a subprocess (`s7_subprocess.jl`)

**① Parent holding CUDA state.**

| Parent state | Free MiB | Build |
|---|---|---|
| no CUDA | 11264 | ok |
| context only | 10669 | ok |
| 9 GiB live | 1453 | ok |
| 10.1 GiB live | 301 | **fails**: `User allocator error allocating 71303168-byte buffer` → `Could not find any implementation for node …` |
| same memory freed, cached in CUDA.jl's pool | 301 | **fails** the same way |
| after `CUDACore.reclaim()` | 10669 | ok |

Memory cached in CUDA.jl's pool is invisible to the child until `reclaim()`
returns it to the driver. TensorRT's message does not say "out of memory".

**② Builder failures.**
- Truncated ONNX: exit 2, parser messages (`Failed to parse the ONNX model`).
- A text file: parses "successfully" into an empty network; the build then
  fails with `Network must have at least one output` (exit 3).
- SIGKILL during the build: Julia sees signal 9; no output file.

**③ `CUDA_VISIBLE_DEVICES`.** Index `0` and the UUID (`GPU-b2b9a022-…`, which
`CUDACore.uuid(device())` reproduces exactly as nvidia-smi prints it) both
work. A bad value makes `createInferBuilder` fail (`CUDA initialization
failure with error: 100`) and return null; the spike, which does not check,
segfaults.

**Conclusions**
- Always `CUDACore.reclaim()` before starting the builder; when a build
  fails, report the free device memory next to the builder's messages.
- The builder checks every TensorRT `create*` result for null and checks
  that the parsed network has at least one input and one output.
- The builder writes to a temporary file in the cache; Julia renames it
  after a successful exit. A killed builder can leave only a temporary file.
- The child gets `CUDA_VISIBLE_DEVICES=GPU-<uuid>` of the Julia task's
  device: absolute, so it is right whatever the parent's own setting is.

## S8 — two processes, one build (`s8_lock.jl`)

`FileWatching.Pidfile.mkpidlock(lock; stale_age = 10)` around "build unless
the product exists", written to a temporary name and renamed:

- Three processes started together: exactly one built (4.8 s); the other
  two waited 4.9 and 5.2 s and found the product. Nothing left behind.
- The lock holder SIGKILLed 2 s into its build: the waiting process removed
  the stale lock after `stale_age` (warning `attempting to remove probably
  stale pidfile`), built, and finished 14.4 s after the kill. The killed
  process left its temporary file (`key.engine.tmp-<pid>`).

The holder's mtime refresh (every `stale_age/2`) runs on a Julia timer; it
fires because the holder only ever waits on subprocesses (g++, the builder).
A dead holder's lock is removed after `stale_age` to `2·stale_age`.

**Conclusions**
- One lock file per cache key, next to the product; `stale_age = 10` s.
- Try the lock first; only when it is taken, print one line: another
  process is building this engine, waiting (with the lock's path).
- While holding the lock, delete leftover temporary files of that key: the
  lock guarantees nobody else is writing them. Temporary names never count
  as products.

## S9 — graph at the smallest batches (`s9_graph.jl`)

Profile 1/14400/40000 (S4's default); graph captured with CUDACore's
`capture`/`instantiate`/`launch`:

| N | enqueueV3 ms | host µs | graph ms | host µs | outputs |
|---|---|---|---|---|---|
| 1 | 0.184 | 108.7 | 0.116 | 2.6 | bitwise |
| 16 | 0.187 | 111.1 | 0.117 | 2.9 | bitwise |
| 64 | 0.186 | 112.9 | 0.119 | 2.6 | bitwise |
| 256 | 0.190 | 109.2 | 0.129 | 2.8 | bitwise |
| 1000 | 0.215 | 110.0 | 0.172 | 2.8 | bitwise |
| 4000 | 0.458 | 114.0 | 0.432 | 4.7 | bitwise |
| 14400 | 1.309 | 196.3 | 1.235 | 5.4 | bitwise |

The graph saves the host submission (≈ 70 µs): −35% below 256 voxels, −20%
at 1000, ≤ 6% from 4000 up.

**Conclusions**
- For MRSTAT's batch sizes (≥ 10⁴ voxels) the graph stays within ≈ 6%, and
  it cannot follow v1's per-call rebinding of caller-owned `out` without a
  re-capture that costs more than it saves. **v1 has no graph**, confirmed.
- The small-batch gain is real but only ≈ 70 µs per call; revisit if a
  small-batch hot loop ever matters.

## S10 — staging the input without allocating (`s10_staging.jl`)

Added 2026-10-01, while implementing task 9: an uncertain point the plan did
not list. The engine reads one (3, N) Float32 matrix [T₁; T₂; B₁]; callers
hand in three vectors, which `predict!` must copy into a staging buffer on
the task's stream, and after warm-up `predict!` must not allocate on the
host. CUDACore 6.4.0, N = 14400:

| Staging | host bytes per call | wall µs | host submission µs |
|---|---|---|---|
| 3 broadcasts `X[k, :] .= v` (MRSTAT's) | 7296 | 26.3 | 20.9 |
| 1 `@cuda` kernel writing all three rows | 912 | 9.5 | 4.7 |
| 3 strided `cuMemcpy2DAsync` (`CUDACore.unsafe_copy2d!`, pitch 4 → 12) | **0** | 46.6 | 10.8 |

All three stage identical values. Wall time against N:

| N | `@cuda` kernel | 3 strided copies |
|---|---|---|
| 1000 | 9.6 µs | 16.3 µs |
| 14400 | 10.3 µs | 46.7 µs |
| 40000 | 10.6 µs | 106.5 µs |

Every Julia kernel launch allocates: CUDACore boxes each argument
(`ArgBox`) and the pointer array (`Ref`) in `pack_arguments`. Avoiding that
would mean launching with CUDACore internals (`KernelState`), which would
break on its upgrades.

**Conclusion.** Stage with three strided copies: the only way that
allocates nothing. They cost ≈ 3% of the engine's own time from 14400 voxels
up (7% at 1000). A `@cuda` kernel would take ≈ 10 µs flat, at 912 bytes per
call; revisit if the zero-allocation requirement is ever relaxed.

## Decisions

Phase 0 settles the open questions below. Where the plan's "Decisions needed"
table recommends an option, implementation follows that recommendation.

| # | Question | Decision | Source |
|---|---|---|---|
| D1 | Bundle location (plan decision 1) | In the package repository, `models/` (ONNX, card, RF tables); sha256 verified at load | plan recommendation |
| D2 | Who writes the card (decision 2) | Hand-written here, with the source (file:line) of every field | plan recommendation |
| D3 | TF32 (decision 3) | Off by default, part of the cache key | S3 + plan |
| D4 | Header source (decision 4) | Deb/rpm `-dev` packages only, including `cuda-cudart-dev` **and `cuda-crt`** | S1 + plan |
| D5 | CUDACore vs CUDA.jl (decision 5) | CUDACore | plan recommendation |
| D6 | Julia compat (decision 6) | 1.12 | plan recommendation |
| D7 | Repository location (decision 7) | Local only; nothing is pushed or published | plan + user instruction |
| D8 | Link libcudart? | No | S1 ② |
| D9 | Version detection | `ccall` into the system libraries, not `nm` | S1 ③ |
| D10 | Default profile | MIN/OPT/MAX = 1/14400/40000; `batch` keyword for others | S4 |
| D11 | Workspace cap | None (TensorRT default) | S5 |
| D12 | Signal-only derivatives | One shared Engine-owned scratch buffer for dT1 and dT2 | S6 ①② |
| D13 | Misaligned `out` | `ArgumentError` (TensorRT documents 256-byte alignment); refines the contract's "any address" | S6 ③ |
| D14 | Before a build | `CUDACore.reclaim()`; report free memory on failure | S7 ① |
| D15 | GPU for the builder | `CUDA_VISIBLE_DEVICES=GPU-<uuid>` | S7 ③ |
| D16 | Lock parameters | Pidfile per key, `stale_age = 10` s, message only when contended | S8 |
| D17 | CUDA graph | Not in v1 | S9 |
| D18 | Tolerances | Bitwise within one engine; 5e-6 relative between builds | S2, S4, S6 |
| D19 | Input staging | Three strided `cuMemcpy2DAsync`: zero host allocation, ≈ 3% of engine time | S10 |
