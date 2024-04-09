/**
 * cub v1.15.1
 * 
*/

#pragma once

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>

#include "common.cuh"

/// cub merge-based CsrMV
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_cub_merge_based(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {
    using namespace cub;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Caching allocator for device memory
    // cub::CachingDeviceAllocator  allocator(true);          

    // Get amount of temporary storage needed
    CubDebugExit(DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   (cudaStream_t)0, false));

    // Allocate
    // CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    checkCudaErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    Timer::kernel_start();
    CubDebugExit(DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, 
                                   const_cast<mat_value_t *>(Ax), 
                                   const_cast<offset_t *>(Ap), 
                                   const_cast<index_t *>(Aj), 
                                   const_cast<vec_x_value_t *>(x), 
                                   y, n_rows, n_cols, nnz,
                                   (cudaStream_t)0, false));
    Timer::kernel_stop();
    checkCudaErr(cudaFree(d_temp_storage));
}