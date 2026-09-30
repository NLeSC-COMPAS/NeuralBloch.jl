// common.h — shared by engine.cpp (the runtime library, libneuralbloch.so)
// and builder.cpp (the builder executable, neuralbloch-build): TensorRT's
// logger, the error slot, versions.
//
// Rules for everything in csrc/:
//   - only the C++17 standard library and TensorRT's public headers;
//   - the exported interface is extern "C" and noexcept, so no C++ exception
//     ever crosses into Julia; everything else has hidden visibility
//     (compiled with -fvisibility=hidden);
//   - TensorRT objects are owned by std::unique_ptr (RAII);
//   - nothing calls back into Julia: TensorRT may log from its own threads.
//
// Nothing here uses the CUDA runtime API, so nothing links libcudart
// (dev/NOTES.md, S1): TensorRT carries its own, statically linked.

#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <cstdio>
#include <string>

#define NB_API extern "C" __attribute__((visibility("default")))

namespace nb {

// The calling thread's last error message, empty when the last call
// succeeded. Julia reads it right after a failed call: a task cannot move to
// another thread between two ccalls without yielding.
inline thread_local std::string last_error;

inline void set_error(const std::string& message) noexcept {
    try {
        last_error = message;
    } catch (...) {  // out of memory while reporting: keep what fits
    }
}

// TensorRT's logger. Errors TensorRT reports are appended to the calling
// thread's error slot, so a failed call returns TensorRT's own explanation;
// messages at or above `echo` also go to stderr.
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity echo) noexcept : echo_(echo) {}

    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            try {
                if (!last_error.empty()) last_error += "\n";
                last_error += message;
            } catch (...) {
            }
        }
        if (severity <= echo_) std::fprintf(stderr, "[TensorRT] %s\n", message);
    }

private:
    Severity echo_;
};

// TensorRT's version as {major, minor, patch, build}: the headers compiled
// against, and the library loaded at run time.
inline void header_version(int32_t* v) noexcept {
    v[0] = NV_TENSORRT_MAJOR;
    v[1] = NV_TENSORRT_MINOR;
    v[2] = NV_TENSORRT_PATCH;
    v[3] = NV_TENSORRT_BUILD;
}

inline void library_version(int32_t* v) noexcept {
    const int32_t packed = getInferLibVersion();  // major·10⁴ + minor·10² + patch
    v[0] = packed / 10000;
    v[1] = packed / 100 % 100;
    v[2] = packed % 100;
    v[3] = getInferLibBuildVersion();
}

}  // namespace nb
