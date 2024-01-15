/**
 * cub v1.15.1
 * 
*/

#pragma once

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

// use texture memory for vector X (get bad performance, not use in cub)
// #define MERGE_USE_TEXTURE

#include <cub/util_allocator.cuh>

#include "common.cuh"
#include "./device_spmv.cuh"

template <typename mat_value_t, 
          typename vec_x_value_t,
          typename vec_y_value_t>
struct MergeFunctor {
    __host__ __device__ __forceinline__
    static vec_y_value_t initialize() {
        return vec_y_value_t{0};
    }

    __host__ __device__ __forceinline__
    static vec_y_value_t combine(const mat_value_t& nonezero, const vec_x_value_t& x) {
        return vec_y_value_t(nonezero * x);
    }

    __host__ __device__ __forceinline__
    static vec_y_value_t reduce(const vec_y_value_t& lhs, const vec_y_value_t& rhs) {
        return lhs + rhs;
    }

};

/// cub merge-based Generalized CsrMV with independent code
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_merge_based_generalized(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    using namespace merge_genl;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Caching allocator for device memory
    cub::CachingDeviceAllocator  allocator(true);      

    using functor_t = MergeFunctor<mat_value_t, vec_x_value_t, vec_y_value_t>;

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV<functor_t>(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   (cudaStream_t)0, false));

    // Allocate
    CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    Timer::kernel_start();
    CubDebugExit(DeviceSpmv::CsrMV<functor_t>(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   (cudaStream_t)0, false));
    Timer::kernel_stop();
}