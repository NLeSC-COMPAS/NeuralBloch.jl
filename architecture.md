# NeuralBloch: how the TensorRT engines are compiled, built and run

NeuralBloch turns hash-pinned ONNX models, one per target (signal and derivatives, signal, coefficients), into TensorRT engines on the user's machine, then runs the one each call asks for on GPU arrays.

## Overview

### Layers

```mermaid
flowchart BT
    L0["L0: csrc/ (C++)<br/>engine.cpp, builder.cpp, common.h<br/>TensorRT calls, C ABI"]
    L1["L1: src/tensorrt/ (Julia)<br/>find system, compile, build, cache, run engines<br/>knows nothing about the model"]
    L2["L2: src/ (Julia)<br/>model bundle, I/O contract, predict!"]
    L0 -->|"compiled and loaded by"| L1
    L1 -->|"used by"| L2
```

### Phases

| | 1. Compile | 2. Build | 3. Inference |
|---|---|---|---|
| Runs | once per C++ source, TensorRT version and g++ | once per ONNX, GPU model, TensorRT version, settings and C++ source | every `predict!` call |
| Entry point | `compile(sys)` | `build(onnx)` | `Handle(path)`, `predict!` |
| Input | `csrc/*`, g++, TensorRT headers | ONNX, GPU, `BuildSettings`, `nb-trt-build` | `model.engine`, `libnb-trtinfer.so`, GPU arrays |
| Output | `libnb-trtinfer.so`, `nb-trt-build` | `model.engine`, one per target | signal, ∂T₁, ∂T₂; signal; or coefficients |
| Cache | `native/` | `engines/` | none |
| Runs as | `g++` subprocess | `nb-trt-build` subprocess | `ccall` into a shared library, in the Julia process |
| Error | `CompileError` (g++ failed), `PrerequisiteError` (header and library versions differ) | `BuildError` | `TensorRTError` |

Both caches live under `cache_root()`; deleting it is always safe, entries are rebuilt on demand.

```mermaid
flowchart LR
    A["1. Compile<br/>g++ on csrc/"] -->|"nb-trt-build"| B["2. Build<br/>ONNX to model.engine"]
    A -->|"libnb-trtinfer.so"| C["3. Inference<br/>predict!"]
    B -->|"model.engine"| C
```

## Files by phase

Only the files that take part in a phase are listed.

### Module files (all phases)

| File | Role |
|---|---|
| `src/tensorrt/TensorRT.jl` | Defines the L1 module; includes `system`, `cache`, `compile`, `build`, `engine` in dependency order. |
| `src/NeuralBloch.jl` | Defines the top-level module; includes `tensorrt/TensorRT.jl`, `model.jl`, `io.jl`, `predict.jl`; sets the public API. |

### 1. Compile

Goal: turn `csrc/` into the two native products.

| File | Role |
|---|---|
| `src/tensorrt/system.jl` | `detect()` finds `libnvinfer`, `libnvonnxparser`, the TensorRT and CUDA headers, g++ and the GPU. `require(sys; gpu = false)` throws a `PrerequisiteError` with install hints if anything but the GPU is missing. |
| `src/tensorrt/compile.jl` | `compile()` runs g++ on both sources, then checks header against library versions with `nb-trt-build --version`. |
| `src/tensorrt/cache.jl` | `native/` entries: hashed key, per-entry lock, temp directory, atomic publish. |
| `csrc/builder.cpp` | Compiled into the `nb-trt-build` executable. |
| `csrc/engine.cpp` | Compiled into `libnb-trtinfer.so`. |
| `csrc/common.h` | Included by both: `nb::Logger`, `last_error`, `nb::fail`, `NB_API`, version helpers. |
| `csrc/exports.map` | Linker version script: only `nb_*` symbols are exported. |
| `src/predict.jl` | `Engine()` triggers it through `detect` and `compile`. |

### 2. Build

Goal: turn each target's ONNX model into a serialized `model.engine` for this GPU. The bundle has three targets (signal and derivatives, signal, coefficients); each is built, cached and opened on its own, and the build layer does not know they belong together.

| File | Role |
|---|---|
| `src/model.jl` | `Model()` reads the `fisp600-rank16` artifact, which `Artifacts.toml` binds and Pkg installs read-only in the depot, or `Model(dir)` any bundle directory. It checks `model_card.toml` (schema 1.x), each target's ONNX sha256 pin and the RF tables, and supplies the ONNX paths. |
| `Artifacts.toml`, `dev/bind_bundle.jl` | The entry of each model family: the git tree hash, sha256 and release URL of its bundle. `bind_bundle.jl` (not package code) writes an entry after `Model` and `contract_problems` accept the release file. |
| `src/predict.jl` | `Engine(model; batch, tf32)` creates the `BuildSettings` and calls `build` for each target, before opening any. |
| `src/tensorrt/system.jl` | `require(sys)` also checks the GPU; supplies name, capability and UUID for the engine key and `CUDA_VISIBLE_DEVICES`. |
| `src/tensorrt/build.jl` | `BuildSettings`, `build()`, and `run_builder` (subprocess, watchdog, exit-code messages, `BuildError`). Before the subprocess starts, `build` returns CUDA.jl's pooled memory to the driver. |
| `src/tensorrt/compile.jl` | Supplies `Native.builder`, the executable `run_builder` runs. |
| `src/tensorrt/cache.jl` | `engines/` entries: `engine_key`, `getorbuild`, manifest and size checks. |
| `csrc/builder.cpp` | Runs as `nb-trt-build`: parses the ONNX, applies TF32, optimization level and batch profile, builds and writes `model.engine`, prints a TOML report, exits with code 1–7. |
| `csrc/common.h` | Logger sends TensorRT errors to stderr, which becomes the `BuildError` message. |

### 3. Inference

Goal: open the engines once, then run the one each call asks for.

**Opening the engines (once per `Engine`)**

| File | Role |
|---|---|
| `src/tensorrt/compile.jl` | Supplies `Native.library`, which `Handle` loads. |
| `src/tensorrt/engine.jl` | `Handle` loads `libnb-trtinfer.so` with `dlopen` and `dlsym`, calls `nb_open`, and reads `TensorInfo` (name, dtype, shape, profile). |
| `csrc/engine.cpp` | `nb_open` deserializes the engine and creates the execution context; `nb_io_*` describe the tensors. |
| `src/io.jl` | `resolve_io` compares the tensors each engine reports (`TensorInfo`) with its target's I/O contract and returns the tensor indices, or a `BundleError`. |
| `src/predict.jl` | `Engine()` makes a `Target` per engine: the handle, the indices and its own `(3, max)` input buffer. It reads the batch sizes from one engine (one `BuildSettings` built all three) and stores `sys.gpu.name`. |

**Every call (`predict!`)**

| File | Role |
|---|---|
| `src/io.jl` | `targetof` (the target the keys of `out` choose, from its type) and `outputlayout` (the rows and element type of that target's outputs); `checkinputs`, `checkoutputs`, `batch_problem`, plus alignment and overlap checks, all before anything is queued. Also the ASCII/Unicode name mapping (`dT1` and `∂T₁`). |
| `src/predict.jl` | `predict!`, `predict` and `outputs`. `predict!` runs the engine of the target `targetof` chooses; `stage!` copies `T₁`, `T₂`, `B₁` into that target's input buffer; `bind!` binds its outputs. `checkdomain` and `close` are also here. |
| `src/model.jl` | Holds the targets (`TARGETS`, which `checktarget` checks names against), `ntr`, `rank`, the training domain and the sequence metadata; `ntr`, `rank`, `domain` and `sequence` in `predict.jl` read them. |
| `src/tensorrt/engine.jl` | `claim!` orders streams, `set_shape!` checks the profile, `set_address!` binds device pointers, `run!` enqueues the run. `close` and the finalizer free the handle. |
| `csrc/engine.cpp` | `nb_set_shape`, `nb_set_address`, `nb_enqueue` (`enqueueV3` on the CUDA stream), `nb_close`. |
| `csrc/common.h` | Logger fills `last_error`, which `nb_last_error()` returns to Julia. |

## Detailed flow

```mermaid
flowchart TD
    subgraph P1["1. COMPILE: once per source / TensorRT / g++"]
        SYS1["system.jl<br/>detect, require (no GPU)"] -->|"compiler path, header dirs,<br/>TensorRT version"| CMP
        CS["csrc/ engine.cpp, builder.cpp,<br/>common.h, exports.map"] -->|"sources, sha256 for key"| CMP["compile.jl<br/>g++ + version check"]
        CMP <-->|"lookup / publish<br/>(native key)"| CA1["cache.jl<br/>native/ entry"]
        CMP -->|"g++ -shared<br/>engine.cpp"| LIB["libnb-trtinfer.so"]
        CMP -->|"g++<br/>builder.cpp"| EXE["nb-trt-build"]
    end

    subgraph P2["2. BUILD: once per ONNX / GPU / TensorRT / settings"]
        MOD["model.jl<br/>Model: ONNX per target + card"] -->|"one ONNX path per target"| ENG1
        ENG1["predict.jl<br/>Engine(model; batch, tf32)"] -->|"each onnx, BuildSettings"| BLD["build.jl<br/>BuildSettings, build"]
        SYS2["system.jl<br/>require + GPU UUID"] -->|"GPU name, capability, UUID"| BLD
        BLD <-->|"lookup / publish<br/>(engine key)"| CA2["cache.jl<br/>engines/ entry"]
        BLD -->|"on cache miss"| RUN["run_builder (subprocess)"]
        EXE -->|"Native.builder path"| RUN
        RUN -->|"--onnx --output --batch<br/>--tf32 --optlevel,<br/>CUDA_VISIBLE_DEVICES=GPU-uuid"| BC["builder.cpp<br/>parse ONNX, profile, build, write"]
        BC -->|"serialized engine file"| PLAN["model.engine"]
        BC -.->|"stdout: TOML report<br/>stderr / exit code"| RUN
    end

    subgraph P3["3. INFERENCE: every call"]
        HND["engine.jl<br/>Handle: dlopen + nb_open"] -->|"ccall nb_open(path)"| EC["engine.cpp<br/>nb_open / nb_io_*"]
        EC -->|"TensorInfo: name, dtype,<br/>shape, profile"| BIND["io.jl<br/>resolve_io: check against the<br/>I/O contract, find indices"]
        BIND -->|"per target: x and its outputs'<br/>tensor indices (or BundleError)"| PRED["predict!(out, engine, p)<br/>keys of out choose the target"]
        PRED -->|"p, out"| CHK["io.jl<br/>targetof, outputlayout,<br/>checkinputs / checkoutputs"]
        CHK -->|"validated N"| STG["predict.jl<br/>stage! into the target's<br/>input buffer"]
        STG -->|"(3, N) Float32 on stream"| CALL["engine.jl<br/>claim! / set_shape! / set_address! / run!"]
        CALL -->|"ccall: shape (N, 3),<br/>device pointers, stream"| EC2["engine.cpp<br/>nb_set_shape / nb_set_address / nb_enqueue"]
        EC2 -->|"enqueueV3(stream)"| GPU["TensorRT on GPU"]
        GPU -.->|"the target's outputs written<br/>into caller's CuMatrix, async"| PRED
    end

    CH["common.h<br/>Logger, last_error"] -.->|"#include"| CS
    LIB -->|"Native.library path"| HND
    PLAN -->|"engine path"| HND
```

Solid arrows carry data or calls; dotted arrows carry results, reports or includes. `nb-trt-build` feeds the build phase and `libnb-trtinfer.so` feeds inference. `model.engine` is only loaded in inference, never rebuilt there.
