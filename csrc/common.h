// common.h — shared by engine.cpp (the runtime library, libnb-trtinfer.so)
// and builder.cpp (the builder executable, nb-trt-build): TensorRT's
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

// Record that `what` failed, followed by whatever TensorRT logged about it
// since the call began (entry points clear the slot first).
inline void fail(const std::string& what) noexcept {
    try {
        last_error = last_error.empty() ? what : what + ": " + last_error;
    } catch (...) {  // out of memory while reporting: keep what is there
    }
}

// TensorRT's logger. Errors are appended to the calling thread's error slot,
// so a failed call reports TensorRT's own explanation; warnings go to stderr,
// and so do errors when `echo_errors` (the builder, whose stderr is its log).
class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(bool echo_errors) noexcept : echo_errors_(echo_errors) {}

    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) {
            try {
                if (!last_error.empty()) last_error += "\n";
                last_error += message;
            } catch (...) {
            }
        }
        if (severity == Severity::kWARNING || (severity <= Severity::kERROR && echo_errors_))
            std::fprintf(stderr, "[TensorRT] %s\n", message);
    }

private:
    bool echo_errors_;
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
