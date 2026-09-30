// Spike helper (2026-09-30): a throwaway runtime library for the Phase 0 spikes.
//
// Questions it serves:
//   S1  does a graph-free runtime library compile against the deb headers
//       alone, and does it link without libcudart?
//   S2+ run any engine (ours or MRSTAT's) from Julia on CuArray memory.
//   S6  what do setTensorAddress / enqueueV3 do with a null output address?
//
// Not package code: no error slot per handle, no careful ABI. Results and
// conclusions live in dev/NOTES.md. Compiled by dev/spikes/common.jl.

#include <NvInferRuntime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Logger : nvinfer1::ILogger {
    char last[1024] = {0};
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR) std::snprintf(last, sizeof last, "%s", msg);
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
} g_logger;

struct Handle {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
};

Handle* H(void* h) { return static_cast<Handle*>(h); }

}  // namespace

extern "C" {

int32_t sp_header_version() noexcept { return NV_TENSORRT_VERSION; }
int32_t sp_lib_version() noexcept { return getInferLibVersion(); }
const char* sp_last_log() noexcept { return g_logger.last; }

void* sp_open(const char* path) noexcept {
    try {
        std::ifstream f(path, std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
        auto h = std::make_unique<Handle>();
        h->runtime.reset(nvinfer1::createInferRuntime(g_logger));
        if (!h->runtime) return nullptr;
        h->engine.reset(h->runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!h->engine) return nullptr;
        h->context.reset(h->engine->createExecutionContext());
        if (!h->context) return nullptr;
        return h.release();
    } catch (...) {
        return nullptr;
    }
}

void sp_close(void* h) noexcept { delete H(h); }

int32_t sp_nb_io(void* h) noexcept { return H(h)->engine->getNbIOTensors(); }
const char* sp_io_name(void* h, int32_t i) noexcept { return H(h)->engine->getIOTensorName(i); }

int64_t sp_device_memory(void* h) noexcept { return H(h)->engine->getDeviceMemorySizeV2(); }

int32_t sp_profile(void* h, const char* name, int32_t sel, int64_t* dims) noexcept {
    auto d = H(h)->engine->getProfileShape(name, 0, static_cast<nvinfer1::OptProfileSelector>(sel));
    for (int32_t i = 0; i < d.nbDims; ++i) dims[i] = d.d[i];
    return d.nbDims;
}

int32_t sp_set_shape(void* h, const char* name, const int64_t* dims, int32_t nb) noexcept {
    nvinfer1::Dims d;
    d.nbDims = nb;
    for (int32_t i = 0; i < nb; ++i) d.d[i] = dims[i];
    return H(h)->context->setInputShape(name, d) ? 1 : 0;
}

int32_t sp_set_address(void* h, const char* name, void* p) noexcept {
    return H(h)->context->setTensorAddress(name, p) ? 1 : 0;
}

int32_t sp_enqueue(void* h, void* stream) noexcept {
    return H(h)->context->enqueueV3(static_cast<cudaStream_t>(stream)) ? 1 : 0;
}

}  // extern "C"
