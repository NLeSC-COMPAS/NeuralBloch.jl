// engine.cpp — libnb-trtinfer.so, the runtime side: Julia ccalls into it to
// open serialized engines and run them. No CUDA graph (dev/NOTES.md, S9).
//
// A handle owns one engine and one execution context. The context allocates
// its scratch memory when created (TensorRT's default, sized for the
// profile's MAX), outside CUDA.jl's pool.
//
// Every entry point is noexcept. Failures return nullptr / -1 and leave the
// reason, with TensorRT's own messages, for nb_last_error(). Tensors are
// addressed by index, 0 ≤ i < nb_io_count(h), so the hot path passes no
// strings. Julia checks indices; the checks here only keep a bad one from
// crashing.
//
// TensorRT uses the CUDA context current on the calling thread: Julia makes
// CUDA.jl's context current first, so raw CuArray pointers bind directly.

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "common.h"

using namespace nvinfer1;

namespace {

nb::Logger logger(false);  // errors go to Julia through nb_last_error(); outlives every runtime

struct Handle {
    // Destroyed in reverse order: context, engine, runtime.
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    std::vector<std::string> names;  // I/O tensor names, by index
};

Handle* handle(void* h) { return static_cast<Handle*>(h); }

const char* name(void* h, int32_t i) {
    const auto& names = handle(h)->names;
    return i >= 0 && i < static_cast<int32_t>(names.size()) ? names[i].c_str() : nullptr;
}

// Copy `d` into dims (room for Dims::MAX_DIMS); its rank, or -1 if invalid.
int32_t copy(const Dims& d, int64_t* dims) {
    if (d.nbDims < 0) return -1;
    for (int32_t k = 0; k < d.nbDims; ++k) dims[k] = d.d[k];
    return d.nbDims;
}

}  // namespace

// {major, minor, patch, build} of the TensorRT headers this library was
// compiled against, and of the TensorRT library it runs with.
NB_API void nb_header_version(int32_t* v) noexcept { nb::header_version(v); }
NB_API void nb_library_version(int32_t* v) noexcept { nb::library_version(v); }

// The calling thread's last error message ("" after a success).
NB_API const char* nb_last_error() noexcept { return nb::last_error.c_str(); }

// Deserialize the engine in `path` and create its execution context.
NB_API void* nb_open(const char* path) noexcept {
    nb::last_error.clear();
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            nb::fail(std::string("cannot read ") + path);
            return nullptr;
        }
        const std::vector<char> plan((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto h = std::make_unique<Handle>();
        h->runtime.reset(createInferRuntime(logger));
        if (!h->runtime) {
            nb::fail("TensorRT could not create a runtime");
            return nullptr;
        }
        h->engine.reset(h->runtime->deserializeCudaEngine(plan.data(), plan.size()));
        if (!h->engine) {
            nb::fail(std::string("cannot deserialize ") + path);
            return nullptr;
        }
        h->context.reset(h->engine->createExecutionContext());
        if (!h->context) {
            nb::fail("cannot create an execution context (out of GPU memory?)");
            return nullptr;
        }
        for (int32_t i = 0; i < h->engine->getNbIOTensors(); ++i)
            h->names.emplace_back(h->engine->getIOTensorName(i));
        return h.release();
    } catch (const std::exception& e) {
        nb::fail(std::string("unexpected C++ exception: ") + e.what());
    } catch (...) {
        nb::fail("unexpected C++ exception");
    }
    return nullptr;
}

NB_API void nb_close(void* h) noexcept { delete handle(h); }

// --- I/O description ------------------------------------------------------------

NB_API int32_t nb_io_count(void* h) noexcept { return static_cast<int32_t>(handle(h)->names.size()); }

NB_API const char* nb_io_name(void* h, int32_t i) noexcept { return name(h, i); }

NB_API int32_t nb_io_is_input(void* h, int32_t i) noexcept {
    const char* n = name(h, i);
    return n ? handle(h)->engine->getTensorIOMode(n) == TensorIOMode::kINPUT : -1;
}

// TensorRT's DataType of tensor i, as its enum value.
NB_API int32_t nb_io_dtype(void* h, int32_t i) noexcept {
    const char* n = name(h, i);
    return n ? static_cast<int32_t>(handle(h)->engine->getTensorDataType(n)) : -1;
}

// The engine's shape of tensor i, TensorRT's (row-major) order, -1 for a
// dynamic dimension; returns the rank.
NB_API int32_t nb_io_shape(void* h, int32_t i, int64_t* dims) noexcept {
    const char* n = name(h, i);
    return n ? copy(handle(h)->engine->getTensorShape(n), dims) : -1;
}

// The shape input i has at the profile's MIN (which = 0), OPT (1) or MAX (2);
// returns the rank, -1 for an output.
NB_API int32_t nb_io_profile(void* h, int32_t i, int32_t which, int64_t* dims) noexcept {
    const char* n = name(h, i);
    if (!n || which < 0 || which > 2) return -1;
    return copy(handle(h)->engine->getProfileShape(n, 0, static_cast<OptProfileSelector>(which)), dims);
}

// Bytes of scratch memory the execution context holds.
NB_API int64_t nb_device_memory(void* h) noexcept { return handle(h)->engine->getDeviceMemorySizeV2(); }

// --- per run ---------------------------------------------------------------------

NB_API int32_t nb_set_shape(void* h, int32_t i, const int64_t* dims, int32_t rank) noexcept {
    nb::last_error.clear();
    const char* n = name(h, i);
    if (!n || rank < 0 || rank > Dims::MAX_DIMS) {
        nb::fail("invalid tensor index or rank");
        return -1;
    }
    Dims d;
    d.nbDims = rank;
    for (int32_t k = 0; k < rank; ++k) d.d[k] = dims[k];
    if (!handle(h)->context->setInputShape(n, d)) {
        nb::fail(std::string("TensorRT rejected the shape of ") + n);
        return -1;
    }
    return 0;
}

NB_API int32_t nb_set_address(void* h, int32_t i, void* address) noexcept {
    nb::last_error.clear();
    const char* n = name(h, i);
    if (!n || !handle(h)->context->setTensorAddress(n, address)) {
        nb::fail(std::string("TensorRT rejected the address of ") + (n ? n : "an invalid tensor index"));
        return -1;
    }
    return 0;
}

// Queue one run on `stream`, with the shapes and addresses set.
NB_API int32_t nb_enqueue(void* h, void* stream) noexcept {
    nb::last_error.clear();
    if (!handle(h)->context->enqueueV3(static_cast<cudaStream_t>(stream))) {
        nb::fail("TensorRT could not enqueue the engine");
        return -1;
    }
    return 0;
}
