
#include <iostream>
#include <cstdio>
#include <numeric>
#include <cmath>

#include "load.hpp"
#include "common.cuh"
#include "spmv/spmv.h"

using namespace std;

using index_t = int;
using offset_t = int;
using value_t = float;

constexpr int TEST_TIMES = 20;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ./bin/<program-name>  <filename.mtx>  <SpMV_kind_string>..." << std::endl;
        exit(1);        
    }

    csr_t<index_t, offset_t, value_t> csr =
        ToCsr(LoadCoo<index_t, offset_t, value_t>(argv[1]));

    vector<string> spmv_kind_strs(argv + 2, argv + argc);

    index_t n_rows, n_cols;
    offset_t nnz;
    n_rows = csr.number_of_rows;
    n_cols = csr.number_of_columns;
    nnz = csr.number_of_nonzeros;

    vector<value_t> vec_x(n_cols, 1);

    vector<value_t> vec_y(n_rows, value_t(0));

    //--------------------------------------------------------------------------
    // Device memory management

    index_t *dA_csrOffsets, *dA_columns;
    value_t *dA_values;
    value_t *dX, *dY;

    CHECK_CUDA(
        cudaMalloc((void **)&dA_csrOffsets, (n_rows + 1) * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, nnz * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, nnz * sizeof(value_t)));

    CHECK_CUDA(cudaMalloc((void **)&dX, n_cols * sizeof(value_t)));
    CHECK_CUDA(cudaMalloc((void **)&dY, n_rows * sizeof(value_t)));

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, csr.row_offsets.data(),
                          (n_rows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, csr.column_indices.data(), nnz * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, csr.nonzero_values.data(), nnz * sizeof(value_t),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(
        cudaMemcpy(dX, vec_x.data(), n_cols * sizeof(value_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(dY, vec_y.data(), n_rows * sizeof(value_t), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // CPU SpMV baseline
    vector<value_t> correct_y(n_rows, 0);
    SpMV_cpu_navie(n_rows, n_cols, nnz, csr.row_offsets.data(),
                   csr.column_indices.data(), csr.nonzero_values.data(),
                   vec_x.data(), correct_y.data());

    printf("Compute delta:\n");
    for (const auto& kind : spmv_kind_strs) {
        //--------------------------------------------------------------------------
        // SpMV APIs
        SpMV(kind, n_rows, n_cols, nnz, dA_csrOffsets, dA_columns, dA_values, dX, dY);
        CHECK_CUDA(cudaMemcpy(vec_y.data(), dY, n_rows * sizeof(value_t), cudaMemcpyDeviceToHost));

        //--------------------------------------------------------------------------
        // device results check
        double delta = 0.;
        for (int i = 0; i < n_rows; ++i) {
            delta += abs(correct_y[i] - vec_y[i]);
        }
        printf("[%-12s] sum: %12lf  avg: %12lf\n", kind.data(), delta, delta / n_rows);
    }
    printf("\n");


    printf("Time cost:\n");
    for (const auto& kind: spmv_kind_strs) {
        //--------------------------------------------------------------------------
        // time cost
        int64_t total_time = 0, kernel_time = 0;
        for (int i = 0; i < TEST_TIMES; ++i) {
            SpMV(kind, n_rows, n_cols, nnz, dA_csrOffsets, dA_columns, dA_values, dX, dY);
            total_time += Timer::total_cost();
            kernel_time += Timer::kernel_cost();
        }
        printf("[%-12s] total: %12lfms  kernel: %12lfms\n", 
            kind.data(), 1. * total_time / TEST_TIMES, 1. * kernel_time / TEST_TIMES);
    }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))

    return EXIT_SUCCESS;
}
