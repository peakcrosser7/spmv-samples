/**
 * cusplibrary v0.4.0
 * 
 * Reference: https://github.com/cusplibrary/cusplibrary/blob/main/cusp/system/cuda/detail/multiply/csr_vector_spmv.h
 * 
*/

#pragma once

#include "./common.cuh"

// use `__syncwarp()` -- CC>=7.0
// #define ENABLE_WARPSYNC

template <unsigned VECTORS_PER_BLOCK, unsigned THREADS_PER_VECTOR,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
__global__ void 
cusp_csr_vector_kernel(index_t n_rows, 
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    __shared__ volatile vec_y_value_t sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile offset_t ptrs[VECTORS_PER_BLOCK][2];

    const offset_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const offset_t thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const offset_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const offset_t vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const offset_t vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const offset_t num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(offset_t row = vector_id; row < n_rows; row += num_vectors) {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
#ifdef ENABLE_WARPSYNC
        __syncwarp();
#endif

        const offset_t row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const offset_t row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        vec_y_value_t sum = (thread_lane == 0) ? initialize(y[row]) : vec_y_value_t(0);

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

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;
#ifdef ENABLE_WARPSYNC
        __syncwarp();
#endif
        // TODO: remove temp var WAR for MSVC    
        vec_y_value_t temp;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) {
            temp = sdata[threadIdx.x + 16];
#ifdef ENABLE_WARPSYNC
            __syncwarp();
            sum = reduce(sum, temp);
            sdata[threadIdx.x] = sum;
            __syncwarp();
#else
            sdata[threadIdx.x] = sum = reduce(sum, temp);
#endif
        }
        if (THREADS_PER_VECTOR >  8) {
            temp = sdata[threadIdx.x +  8];
#ifdef ENABLE_WARPSYNC
            __syncwarp();
            sum = reduce(sum, temp);
            sdata[threadIdx.x] = sum;
            __syncwarp();
#else
            sdata[threadIdx.x] = sum = reduce(sum, temp);
#endif
        }
        if (THREADS_PER_VECTOR >  4) {
            temp = sdata[threadIdx.x +  4];
#ifdef ENABLE_WARPSYNC
            __syncwarp();
            sum = reduce(sum, temp);
            sdata[threadIdx.x] = sum;
            __syncwarp();
#else
            sdata[threadIdx.x] = sum = reduce(sum, temp);
#endif
        }
        if (THREADS_PER_VECTOR >  2) {
            temp = sdata[threadIdx.x +  2];
#ifdef ENABLE_WARPSYNC
            __syncwarp();
            sum = reduce(sum, temp);
            sdata[threadIdx.x] = sum;
            __syncwarp();
#else
            sdata[threadIdx.x] = sum = reduce(sum, temp);
#endif
        }
        if (THREADS_PER_VECTOR >  1) {
            temp = sdata[threadIdx.x +  1];
#ifdef ENABLE_WARPSYNC
            __syncwarp();
            sum = reduce(sum, temp);
            sdata[threadIdx.x] = sum;
            __syncwarp();
#else
            sdata[threadIdx.x] = sum = reduce(sum, temp);
#endif
        }

        // first thread writes the result
        if (thread_lane == 0)
            y[row] = vec_y_value_t(sdata[threadIdx.x]);
    }
}

template <typename Size1, typename Size2>
__host__ __device__
Size1 DIVIDE_INTO(Size1 N, Size2 granularity) {
    return (N + (granularity - 1)) / granularity;
}

template <unsigned THREADS_PER_VECTOR,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
void __cusp_csr_vector(index_t n_rows, 
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    constexpr unsigned THREADS_PER_BLOCK = 128;
    constexpr unsigned VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

#ifdef USE_CUSP_NUM_BLOCKS
    const size_t MAX_BLOCKS = max_active_blocks(
        cusp_csr_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR, index_t,
                               offset_t, mat_value_t, vec_x_value_t,
                               vec_y_value_t, unary_func_t, binary_func1_t,
                               binary_func2_t>,
        THREADS_PER_BLOCK, (size_t)0);
    const size_t NUM_BLOCKS = std::min<size_t>(
        MAX_BLOCKS, DIVIDE_INTO(n_rows, VECTORS_PER_BLOCK));
#else
     const size_t NUM_BLOCKS = std::max<size_t>(1, DIVIDE_INTO(n_rows, VECTORS_PER_BLOCK));
#endif
    
    cusp_csr_vector_kernel<VECTORS_PER_BLOCK, THREADS_PER_VECTOR><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
}

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t,
          typename unary_func_t, typename binary_func1_t, typename binary_func2_t>
void cusp_csr_vector(index_t n_rows, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y,
    unary_func_t initialize, binary_func1_t combine, binary_func2_t reduce) {

    const offset_t nnz_per_row = nnz / n_rows;

    if (nnz_per_row <=  2) {
        Timer::kernel_start();
        __cusp_csr_vector<2>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        cudaDeviceSynchronize();
        Timer::kernel_stop();
        return;
    }
    if (nnz_per_row <=  4) {
        Timer::kernel_start();
        __cusp_csr_vector<4>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        cudaDeviceSynchronize();
        Timer::kernel_stop();
        return;
    }
    if (nnz_per_row <=  8) {
        Timer::kernel_start();
        __cusp_csr_vector<8>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        cudaDeviceSynchronize();
        Timer::kernel_stop();
        return;
    }
    if (nnz_per_row <= 16) {
        Timer::kernel_start();
        __cusp_csr_vector<16>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
        cudaDeviceSynchronize();
        Timer::kernel_stop();
        return;
    }

    Timer::kernel_start();
    __cusp_csr_vector<32>(n_rows, Ap, Aj, Ax, x, y, initialize, combine, reduce);
    cudaDeviceSynchronize();
    Timer::kernel_stop();
}

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_cusp_origin(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    auto initialize = ConstFunctor<vec_y_value_t>(0);
    auto combine = MultFunctor<mat_value_t>();
    auto reduce = PlusFunctor<mat_value_t>();

    cusp_csr_vector(n_rows, nnz, Ap, Aj, Ax, x, y, initialize, combine, reduce);
}