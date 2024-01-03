#pragma once

#include <cuda_runtime.h>

#include "timer.hpp"


#define USED_DEVICE 0

#define FULL_MASK 0xffffffff


template <typename T>
void CheckCudaErr(T result, char const *const func, const char *const file,
                  int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result),
                func);
        abort();
    }
}
#define checkCudaErr(val) CheckCudaErr((val), #val, __FILE__, __LINE__)