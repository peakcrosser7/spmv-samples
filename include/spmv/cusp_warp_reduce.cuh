#pragma once

#include "common.cuh"
#include "./cusp.cuh"

#define FULL_MASK 0xffffffff

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

template <unsigned VECTORS_PER_BLOCK, unsigned THREADS_PER_VECTOR,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
__global__ void 
cusp_warp_reduce_kernel(index_t n_rows, 
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    constexpr offset_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const offset_t thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const offset_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const offset_t row_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    
    if (row_id < n_rows) {        
        offset_t row_start = Ap[row_id];
        offset_t row_end = Ap[row_id + 1];

        // initialize local sum
        vec_y_value_t sum = (thread_lane == 0) ? initialize(y[row_id]) : vec_y_value_t(0);

        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32) {
            // ensure aligned memory access to Aj and Ax

            offset_t jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));
        } else {
            // accumulate local sums
            for(offset_t jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum = reduce(sum, combine(Ax[jj], x[Aj[jj]]));
        }

        // reduce local sums to row sum
        sum = WarpReduceSum<THREADS_PER_VECTOR>(sum);

        // first thread writes the result
        if (thread_lane == 0) {
            y[row_id] = sum;
        }
    }
}

template <unsigned THREADS_PER_VECTOR,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
void __cusp_warp_reduce(index_t n_rows, 
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    constexpr unsigned THREADS_PER_BLOCK = 128;
    constexpr unsigned VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
    const unsigned NUM_BLOCKS = std::max<int>(1, DIVIDE_INTO(n_rows, VECTORS_PER_BLOCK));

    cusp_warp_reduce_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
}

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
void cusp_warp_reduce(index_t n_rows, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    const offset_t nnz_per_row = nnz / n_rows;

    if (nnz_per_row <=  2) {
        __cusp_warp_reduce<2>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <=  4) {
        __cusp_warp_reduce<4>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <=  8) {
        __cusp_warp_reduce<8>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        return;
    }
    if (nnz_per_row <= 16) {
        __cusp_warp_reduce<16>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        return;
    }

    Timer::kernel_start();
    __cusp_warp_reduce<32>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
    cudaDeviceSynchronize();
    Timer::kernel_stop();
}

// use warp shuffle instructions to reduce
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_cusp_warp_reduce(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    auto initialize = ConstFunctor<vec_y_value_t>(0);
    auto combine = MultFunctor<mat_value_t>();
    auto reduce = PlusFunctor<mat_value_t>();

    cusp_warp_reduce(n_rows, nnz, Ap, Aj, Ax, x, y, initialize, combine, reduce);
}