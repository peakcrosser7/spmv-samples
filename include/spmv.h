#pragma once

#include <iostream>
#include <string>

#include "common.cuh"
#include "spmv/cpu_navie.hpp"
#include "spmv/cusparse.cuh"
#include "spmv/cusp/cusp.cuh"
#include "spmv/cusp/cusp_warp_reduce.cuh"
#include "spmv/cusp/cusp_warp_read_reduce.cuh"
#include "spmv/LightSpMV.cuh"
#include "spmv/cub_merge.cuh"

/// SPMV kind strings and its function
#define SPMV_KINDS                                                             \
    X("cusparse", SpMV_cusparse)                                               \
    X("cusp", SpMV_cusp_origin)                                                \
    X("cusp1", SpMV_cusp_warp_reduce)                                          \
    X("cusp2", SpMV_cusp_warp_read_reduce)                                     \
    X("light_vec", SpMV_light_vector)                                          \
    X("light_warp", SpMV_light_warp)                                           \
    X("cub_merge", SpMV_cub_merge_based)

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV(const std::string& kind_str,
    index_t n_rows,  index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    #define X(a, b)                                   \
        if (kind_str == a) {                          \
            Timer::total_start();                     \
            b(n_rows, n_cols, nnz, Ap, Aj, Ax, x, y); \
            Timer::total_stop();                      \
            return;                                   \
        }
    SPMV_KINDS
    #undef X
    
    std::cerr << "SpMV kind \"" << kind_str << "\" is NOT SUPPROT\n";
    exit(EXIT_FAILURE);
}