#pragma once

#include <iostream>

/// CUDA error check macro, which exists on failure
/// https://stackoverflow.com/a/14038590
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        std::exit(1);
    }
}
