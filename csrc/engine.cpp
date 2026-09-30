// engine.cpp — libneuralbloch.so, the runtime side: Julia ccalls into it to
// run engines. No CUDA graph (dev/NOTES.md, S9).

#include "common.h"

// {major, minor, patch, build} of the TensorRT headers this library was
// compiled against, and of the TensorRT library it runs with.
NB_API void nb_header_version(int32_t* v) noexcept { nb::header_version(v); }
NB_API void nb_library_version(int32_t* v) noexcept { nb::library_version(v); }

// The calling thread's last error message ("" after a success).
NB_API const char* nb_last_error() noexcept { return nb::last_error.c_str(); }
