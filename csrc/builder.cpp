// builder.cpp — neuralbloch-build, a standalone executable that turns an ONNX
// model into a serialized TensorRT engine. Julia runs it as a subprocess, so a
// crash or a hang in TensorRT's builder cannot take the Julia session down.
//
// stdout carries a TOML report for Julia to parse; stderr carries TensorRT's
// log.
//
//     neuralbloch-build --version

#include <NvOnnxParser.h>

#include <cstring>

#include "common.h"

namespace {

void print_versions() {
    int32_t h[4], l[4];
    nb::header_version(h);
    nb::library_version(l);
    const int32_t p = getNvOnnxParserVersion();
    std::printf("tensorrt_header = \"%d.%d.%d.%d\"\n", h[0], h[1], h[2], h[3]);
    std::printf("tensorrt_library = \"%d.%d.%d.%d\"\n", l[0], l[1], l[2], l[3]);
    std::printf("onnx_parser = \"%d.%d.%d\"\n", p / 10000, p / 100 % 100, p % 100);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        print_versions();
        return 0;
    }
    std::fprintf(stderr, "usage: %s --version\n", argv[0]);
    return 2;
}
