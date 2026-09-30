// builder.cpp — neuralbloch-build, a standalone executable that turns an ONNX
// model into a serialized TensorRT engine. Julia runs it as a subprocess, so a
// crash or a hang in TensorRT's builder cannot take the Julia session down.
//
//     neuralbloch-build --onnx FILE --output FILE --batch MIN,OPT,MAX --tf32 0|1 --optlevel N
//     neuralbloch-build --version
//
// Every build setting is applied explicitly, never left to TensorRT's
// defaults, and read back into the report:
//   - strongly typed: every tensor keeps the type the ONNX gives it (FP32).
//     Always on since TensorRT 11, which deprecates the flag; set for 10;
//   - TF32 on or off (TensorRT's default is on);
//   - the builder optimization level;
//   - one optimization profile, MIN/OPT/MAX on the batch dimension (0) of
//     every input. Other dimensions must be static.
// The workspace limit stays TensorRT's default, the whole device: a smaller
// one only makes builds fail (dev/NOTES.md, S5).
//
// stdout carries a TOML report of what was built, printed on success only;
// stderr carries TensorRT's warnings and errors, and ours. The GPU is the
// only one CUDA_VISIBLE_DEVICES leaves visible.
//
// Exit codes: 0 success, 1 bad arguments, 2 the ONNX parser rejected the
// model, 3 the network cannot be built this way, 4 TensorRT's build failed,
// 5 the engine could not be written, 6 TensorRT could not start, 7 an
// unexpected C++ exception.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"

using namespace nvinfer1;

namespace {

struct Options {
    std::string onnx, output;
    int64_t batch[3] = {0, 0, 0};
    int tf32 = -1;
    int optlevel = -1;
};

int fail(int code, const char* format, ...) {
    std::fprintf(stderr, "neuralbloch-build: ");
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    return code;
}

bool parse_options(int argc, char** argv, Options& o) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        const char* value = argv[i + 1];
        if (flag == "--onnx") {
            o.onnx = value;
        } else if (flag == "--output") {
            o.output = value;
        } else if (flag == "--batch") {
            long long lo, opt, hi;
            char tail;
            if (std::sscanf(value, "%lld,%lld,%lld%c", &lo, &opt, &hi, &tail) != 3) return false;
            o.batch[0] = lo, o.batch[1] = opt, o.batch[2] = hi;
        } else if (flag == "--tf32") {
            o.tf32 = std::atoi(value);
        } else if (flag == "--optlevel") {
            o.optlevel = std::atoi(value);
        } else {
            return false;
        }
    }
    return argc % 2 == 1 && !o.onnx.empty() && !o.output.empty() && 1 <= o.batch[0] &&
           o.batch[0] <= o.batch[1] && o.batch[1] <= o.batch[2] && (o.tf32 == 0 || o.tf32 == 1) &&
           o.optlevel >= 0;
}

// A TOML basic string.
std::string quoted(const char* s) {
    std::string out = "\"";
    for (; *s; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

std::string names(const std::vector<ITensor*>& tensors) {
    std::string out = "[";
    for (size_t i = 0; i < tensors.size(); ++i) out += (i ? ", " : "") + quoted(tensors[i]->getName());
    return out + "]";
}

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void print_versions() {
    int32_t h[4], l[4];
    nb::header_version(h);
    nb::library_version(l);
    const int32_t p = getNvOnnxParserVersion();
    std::printf("tensorrt_header = \"%d.%d.%d.%d\"\n", h[0], h[1], h[2], h[3]);
    std::printf("tensorrt_library = \"%d.%d.%d.%d\"\n", l[0], l[1], l[2], l[3]);
    std::printf("onnx_parser = \"%d.%d.%d\"\n", p / 10000, p / 100 % 100, p % 100);
}

int build(const Options& o) {
    nb::Logger logger(true);  // stderr is the builder's log: errors too
    const auto t0 = std::chrono::steady_clock::now();

    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) {
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        return fail(6, "TensorRT could not create a builder: no usable GPU (CUDA_VISIBLE_DEVICES=%s)?",
                    visible ? visible : "<unset>");
    }
#if NV_TENSORRT_MAJOR < 11
    const auto flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
    const uint32_t flags = 0;  // strongly typed is the only mode
#endif
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) return fail(6, "TensorRT could not create a network definition");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) return fail(6, "TensorRT could not create an ONNX parser");

    if (!parser->parseFromFile(o.onnx.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::fprintf(stderr, "ONNX parser: %s\n", parser->getError(i)->desc());
        return fail(2, "cannot parse %s", o.onnx.c_str());
    }
    const double parse_seconds = since(t0);
    std::vector<ITensor*> inputs, outputs;
    for (int i = 0; i < network->getNbInputs(); ++i) inputs.push_back(network->getInput(i));
    for (int i = 0; i < network->getNbOutputs(); ++i) outputs.push_back(network->getOutput(i));
    if (inputs.empty() || outputs.empty())
        return fail(3, "%s parses into a network with no inputs or outputs: is it an ONNX model?",
                    o.onnx.c_str());

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return fail(6, "TensorRT could not create a builder configuration");
    if (o.tf32)
        config->setFlag(BuilderFlag::kTF32);
    else
        config->clearFlag(BuilderFlag::kTF32);
    config->setBuilderOptimizationLevel(o.optlevel);

    IOptimizationProfile* profile = builder->createOptimizationProfile();  // owned by the builder
    if (!profile) return fail(6, "TensorRT could not create an optimization profile");
    for (ITensor* in : inputs) {
        const Dims d = in->getDimensions();
        Dims lo = d, opt = d, hi = d;
        for (int k = 0; k < d.nbDims; ++k) {
            if (d.d[k] >= 0) continue;
            if (k != 0)
                return fail(3, "input %s has a dynamic dimension %d; only the batch dimension (0) may be dynamic",
                            in->getName(), k);
            lo.d[0] = o.batch[0], opt.d[0] = o.batch[1], hi.d[0] = o.batch[2];
        }
        if (!profile->setDimensions(in->getName(), OptProfileSelector::kMIN, lo) ||
            !profile->setDimensions(in->getName(), OptProfileSelector::kOPT, opt) ||
            !profile->setDimensions(in->getName(), OptProfileSelector::kMAX, hi))
            return fail(3, "TensorRT rejected the batch range %lld,%lld,%lld for input %s",
                        static_cast<long long>(o.batch[0]), static_cast<long long>(o.batch[1]),
                        static_cast<long long>(o.batch[2]), in->getName());
    }
    if (config->addOptimizationProfile(profile) < 0) return fail(3, "TensorRT rejected the optimization profile");

    const auto t1 = std::chrono::steady_clock::now();
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return fail(4, "TensorRT could not build an engine from %s", o.onnx.c_str());
    const double build_seconds = since(t1);

    std::ofstream file(o.output, std::ios::binary);
    file.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    file.close();
    if (!file) return fail(5, "cannot write %s", o.output.c_str());

#if NV_TENSORRT_MAJOR < 11
    const bool strongly_typed = network->getFlag(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
    const bool strongly_typed = true;
#endif
    print_versions();
    std::printf("strongly_typed = %s\n", strongly_typed ? "true" : "false");
    std::printf("tf32 = %s\n", config->getFlag(BuilderFlag::kTF32) ? "true" : "false");
    std::printf("optlevel = %d\n", config->getBuilderOptimizationLevel());
    std::printf("workspace_bytes = %zu\n", config->getMemoryPoolLimit(MemoryPoolType::kWORKSPACE));
    std::printf("batch = [%lld, %lld, %lld]\n", static_cast<long long>(o.batch[0]),
                static_cast<long long>(o.batch[1]), static_cast<long long>(o.batch[2]));
    std::printf("inputs = %s\noutputs = %s\n", names(inputs).c_str(), names(outputs).c_str());
    std::printf("layers = %d\n", network->getNbLayers());
    std::printf("parse_seconds = %.3f\nbuild_seconds = %.3f\n", parse_seconds, build_seconds);
    std::printf("engine_bytes = %zu\n", plan->size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        print_versions();
        return 0;
    }
    Options options;
    if (!parse_options(argc, argv, options))
        return fail(1, "usage: %s --onnx FILE --output FILE --batch MIN,OPT,MAX --tf32 0|1 --optlevel N"
                       " (with 1 <= MIN <= OPT <= MAX), or --version", argv[0]);
    try {
        return build(options);
    } catch (const std::exception& e) {
        return fail(7, "unexpected C++ exception: %s", e.what());
    } catch (...) {
        return fail(7, "unexpected C++ exception");
    }
}
