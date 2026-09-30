// Spike helper (2026-09-30): a throwaway ONNX -> engine builder for Phase 0.
//
//   spike_builder ONNX OUT MIN OPT MAX TF32 OPTLEVEL WORKSPACE_MIB
//
// TF32 and OPTLEVEL take -1 for "leave TensorRT's default"; WORKSPACE_MIB 0
// does the same. Prints the defaults it found, what it set, and the wall time
// of parse / build / write, one `key = value` per line on stdout, so the
// Julia drivers can parse it. Exit codes: 1 usage, 2 parse, 3 build, 4 write.
//
// Questions it serves: S1 (link: needs no libcudart?), S2 (upstream-equal
// settings), S3 (TF32), S4 (profiles), S5 (build time, workspace), S7 (as a
// subprocess). Not package code; see dev/NOTES.md.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

using namespace nvinfer1;

namespace {

struct Logger : ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
} g_logger;

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s ONNX OUT MIN OPT MAX TF32 OPTLEVEL WORKSPACE_MIB\n", argv[0]);
        return 1;
    }
    const char* onnx = argv[1];
    const char* out = argv[2];
    const int64_t bmin = std::atoll(argv[3]), bopt = std::atoll(argv[4]), bmax = std::atoll(argv[5]);
    const int tf32 = std::atoi(argv[6]), level = std::atoi(argv[7]);
    const long long ws_mib = std::atoll(argv[8]);

    std::printf("header_version = %ld\nlib_version = %d\nparser_version = %d\n",
                long(NV_TENSORRT_VERSION), getInferLibVersion(), getNvOnnxParserVersion());

    auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<IBuilder> builder(createInferBuilder(g_logger));
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, g_logger));
    if (!parser->parseFromFile(onnx, static_cast<int>(ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::fprintf(stderr, "parse error %d: %s\n", i, parser->getError(i)->desc());
        return 2;
    }
    std::printf("parse_s = %.3f\n", since(t0));

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    std::printf("default_tf32 = %d\n", config->getFlag(BuilderFlag::kTF32) ? 1 : 0);
    std::printf("default_optlevel = %d\n", config->getBuilderOptimizationLevel());
    std::printf("default_workspace = %zu\n", config->getMemoryPoolLimit(MemoryPoolType::kWORKSPACE));

    if (tf32 == 0) config->clearFlag(BuilderFlag::kTF32);
    if (tf32 == 1) config->setFlag(BuilderFlag::kTF32);
    if (level >= 0) config->setBuilderOptimizationLevel(level);
    if (ws_mib > 0) config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, size_t(ws_mib) << 20);

    // Every network input gets the batch range on its dynamic (-1) dimensions.
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); ++i) {
        ITensor* in = network->getInput(i);
        Dims d = in->getDimensions();
        Dims lo = d, mid = d, hi = d;
        for (int k = 0; k < d.nbDims; ++k)
            if (d.d[k] < 0) { lo.d[k] = bmin; mid.d[k] = bopt; hi.d[k] = bmax; }
        profile->setDimensions(in->getName(), OptProfileSelector::kMIN, lo);
        profile->setDimensions(in->getName(), OptProfileSelector::kOPT, mid);
        profile->setDimensions(in->getName(), OptProfileSelector::kMAX, hi);
    }
    config->addOptimizationProfile(profile);

    std::printf("tf32 = %d\noptlevel = %d\nworkspace = %zu\n",
                config->getFlag(BuilderFlag::kTF32) ? 1 : 0,
                config->getBuilderOptimizationLevel(),
                config->getMemoryPoolLimit(MemoryPoolType::kWORKSPACE));
    std::fflush(stdout);

    auto t1 = std::chrono::steady_clock::now();
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return 3;
    std::printf("build_s = %.3f\n", since(t1));

    std::ofstream f(out, std::ios::binary);
    f.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    if (!f) return 4;
    std::printf("bytes = %zu\ntotal_s = %.3f\n", plan->size(), since(t0));
    return 0;
}
