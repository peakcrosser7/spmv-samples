#pragma once

#include "common.cuh"


template <typename T>
class ConstFunctor {
public:
    __host__ __device__ __forceinline__ 
    ConstFunctor(const T val = 0) : val(val) {}

    __device__ __forceinline__
    T operator()(const T &x) const { return val; }

private:
    T val;
};

template <typename T>
struct PlusFunctor {
    __device__ __forceinline__
    constexpr T operator()(const T &lhs, const T &rhs) const {
        return lhs + rhs;
    }
};


template <typename T>
struct MultFunctor {
    __device__ __forceinline__
    constexpr T operator()(const T &lhs, const T &rhs) const {
        return lhs * rhs;
    }
};



template <unsigned size, typename T>
__device__ __forceinline__
T WarpReduceSum(T sum) {
    if constexpr (size >= 32) sum += __shfl_down_sync(FULL_MASK, sum, 16); // 0-16, 1-17, 2-18, etc.
    if constexpr (size >= 16) sum += __shfl_down_sync(FULL_MASK, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if constexpr (size >= 8)  sum += __shfl_down_sync(FULL_MASK, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if constexpr (size >= 4)  sum += __shfl_down_sync(FULL_MASK, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if constexpr (size >= 2)  sum += __shfl_down_sync(FULL_MASK, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;   
}

#define CUSP_DEVICE 0

/// use `max_active_blocks()` function in cusp to get NUM_BLOCKS
/// but got bad performance
// #define USE_CUSP_NUM_BLOCKS

//---------------------------------------------------------------------------------
// Copyed from cusplibrary

inline size_t max_blocks_per_multiprocessor(const cudaDeviceProp &properties) {
    int major = properties.major;
    int minor = properties.minor;

    // Reference:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability
    if (major == 9) {
        if (minor == 0) {
            return 32;
        }
        goto not_sup;
    }
    if (major == 8) {
        if (minor == 0) {
            return 32;
        }
        if (minor == 6 || minor == 7) {
            return 16;
        }
        if (minor == 9) {
            return 24;
        }
        goto not_sup;
    }
    if (major == 7) {
        if (minor == 0 || minor == 2) {
            return 32;
        }
        if (minor == 5) {
            return 16;
        }
        goto not_sup;
    }
    if (major == 6) {
        if (minor == 0 || minor == 1 || minor == 2) {
            return 32;
        }
        goto not_sup;
    }
    if (major == 5) {
        if (minor == 0 || minor == 2 || minor == 3) {
            return 32;
        }
        goto not_sup;
    }
    // Reference: 
    // https://github.com/cusplibrary/cusplibrary/blob/bce60ca9c21fe6742ae38baff494c6968d54d372/cusp/system/cuda/detail/cuda_launch_config.h#L248
    if (major > 2) {
        return 16;
    }
    return 8;

not_sup:
    printf("compute capability: %d.%d not support in \"%s\"\n", major, minor, "max_blocks_per_multiprocessor");
    exit(EXIT_FAILURE);
}

// granularity of shared memory allocation
inline size_t smem_allocation_unit(const cudaDeviceProp &properties) {
    switch (properties.major) {
    case 1:
        return 512;
    case 2:
        return 128;
    case 3:
        return 256;
    default:
        return 256; // unknown GPU; have to guess
    }
}

// granularity of register allocation
inline size_t reg_allocation_unit(const cudaDeviceProp &properties,
                                  const size_t regsPerThread) {
    switch (properties.major) {
    case 1:
        return (properties.minor <= 1) ? 256 : 512;
    case 2:
        switch (regsPerThread) {
        case 21:
        case 22:
        case 29:
        case 30:
        case 37:
        case 38:
        case 45:
        case 46:
            return 128;
        default:
            return 64;
        }
    case 3:
        return 256;
    default:
        return 256; // unknown GPU; have to guess
    }
}

// granularity of warp allocation
inline size_t warp_allocation_multiple(const cudaDeviceProp &properties) {
    return (properties.major <= 1) ? 2 : 1;
}

// number of "sides" into which the multiprocessor is partitioned
inline size_t num_sides_per_multiprocessor(const cudaDeviceProp &properties) {
    switch (properties.major) {
    case 1:
        return 1;
    case 2:
        return 2;
    case 3:
        return 4;
    default:
        return 4; // unknown GPU; have to guess
    }
}

// x/y rounding towards +infinity for integers, used to determine # of blocks/warps etc.
template <typename L, typename R>
L divide_ri(const L x, const R y) {
    return (x + (y - 1)) / y;
}

// round x towards infinity to the next multiple of y
template<typename L, typename R>
L round_i(const L x, const R y) {
    return y * divide_ri(x, y);
}

template <typename T>
T min_(const T &lhs, const T &rhs) {
    return rhs < lhs ? rhs : lhs;
}

inline size_t
max_active_blocks_per_multiprocessor(const cudaDeviceProp &properties,
                                     const cudaFuncAttributes &attributes,
                                     int cta_size, size_t dynamic_smem_bytes) {
    // Determine the maximum number of CTAs that can be run simultaneously per
    // SM This is equivalent to the calculation done in the CUDA Occupancy
    // Calculator spreadsheet

    //////////////////////////////////////////
    // Limits due to threads/SM or blocks/SM
    //////////////////////////////////////////
    const size_t maxThreadsPerSM =
        properties.maxThreadsPerMultiProcessor; // 768, 1024, 1536, etc.
    const size_t maxBlocksPerSM = max_blocks_per_multiprocessor(properties);

    // Calc limits
    const size_t ctaLimitThreads = (cta_size <= properties.maxThreadsPerBlock)
                                       ? maxThreadsPerSM / cta_size : 0;
    const size_t ctaLimitBlocks = maxBlocksPerSM;

    //////////////////////////////////////////
    // Limits due to shared memory/SM
    //////////////////////////////////////////
    const size_t smemAllocationUnit = smem_allocation_unit(properties);
    const size_t smemBytes = attributes.sharedSizeBytes + dynamic_smem_bytes;
    const size_t smemPerCTA = round_i(smemBytes, smemAllocationUnit);

    // Calc limit
    const size_t ctaLimitSMem = smemPerCTA > 0
                                    ? properties.sharedMemPerBlock / smemPerCTA
                                    : maxBlocksPerSM;

    //////////////////////////////////////////
    // Limits due to registers/SM
    //////////////////////////////////////////
    const size_t regAllocationUnit =
        reg_allocation_unit(properties, attributes.numRegs);
    const size_t warpAllocationMultiple = warp_allocation_multiple(properties);
    const size_t numWarps = round_i(divide_ri(cta_size, properties.warpSize),
                                    warpAllocationMultiple);

    // Calc limit
    size_t ctaLimitRegs;
    if (properties.major <= 1) {
        // GPUs of compute capability 1.x allocate registers to CTAs
        // Number of regs per block is regs per thread times number of warps
        // times warp size, rounded up to allocation unit
        const size_t regsPerCTA =
            round_i(attributes.numRegs * properties.warpSize * numWarps,
                    regAllocationUnit);
        ctaLimitRegs = regsPerCTA > 0 ? properties.regsPerBlock / regsPerCTA
                                      : maxBlocksPerSM;
    } else {
        // GPUs of compute capability 2.x and higher allocate registers to warps
        // Number of regs per warp is regs per thread times times warp size,
        // rounded up to allocation unit
        const size_t regsPerWarp = round_i(
            attributes.numRegs * properties.warpSize, regAllocationUnit);
        const size_t numSides = num_sides_per_multiprocessor(properties);
        const size_t numRegsPerSide = properties.regsPerBlock / numSides;
        ctaLimitRegs =
            regsPerWarp > 0
                ? ((numRegsPerSide / regsPerWarp) * numSides) / numWarps
                : maxBlocksPerSM;
    }

    //////////////////////////////////////////
    // Overall limit is min() of limits due to above reasons
    //////////////////////////////////////////
    return min_(ctaLimitRegs,
                min_(ctaLimitSMem, min_(ctaLimitThreads, ctaLimitBlocks)));
}

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t cta_size,
                         const size_t dynamic_smem_bytes) {
    
    cudaFuncAttributes attributes;
    checkCudaErr(cudaFuncGetAttributes(&attributes, kernel));
    cudaDeviceProp properties;
    checkCudaErr(cudaGetDeviceProperties(&properties, USED_DEVICE));

    return properties.multiProcessorCount *
           max_active_blocks_per_multiprocessor(
               properties, attributes, cta_size, dynamic_smem_bytes);
}