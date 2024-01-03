/**
 * cuSPARSE v11.70.5
 * 
*/

#pragma once

#include <cusparse.h>      // cusparseSpMV

#include "common.cuh"


#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    }

template <typename T>
cusparseIndexType_t CusparseIndex = {};
template <>
cusparseIndexType_t CusparseIndex<int> = CUSPARSE_INDEX_32I;

template <typename T>
cudaDataType_t CudaDataType = {};
template <>
cudaDataType_t CudaDataType<float> = CUDA_R_32F;
template <>
cudaDataType_t CudaDataType<double> = CUDA_R_64F;

template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_cusparse(index_t n_rows,  index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    const vec_x_value_t alpha = 1.0f;
    const vec_y_value_t beta = 0.0f;

    // cusparse上下文指针
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    void *dBuffer = NULL;
    size_t bufferSize = 0;
    // `cusparseCreate()`:初始化cusparse库,并在cusparse上下文上创建一个句柄
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    // `cusparseCreateCsr()`:创建CSR格式稀疏矩阵的描述符
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, n_rows, n_cols, nnz, const_cast<offset_t *>(Ap),
        const_cast<index_t *>(Aj), const_cast<mat_value_t *>(Ax),
        CusparseIndex<offset_t>, CusparseIndex<index_t>, CUSPARSE_INDEX_BASE_ZERO,
        CudaDataType<mat_value_t>));

    // Create dense vector X
    // `cusparseCreateDnVec()`:构建稠密向量描述符
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n_cols, const_cast<mat_value_t *>(x), CudaDataType<vec_x_value_t>));
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n_rows, y, CudaDataType<vec_y_value_t>));
    
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CudaDataType<vec_y_value_t>, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    checkCudaErr(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMV
    Timer::kernel_start();
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CudaDataType<vec_y_value_t>,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    Timer::kernel_stop();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    // `cusparseDestroy()`:释放CPU侧资源
    CHECK_CUSPARSE(cusparseDestroy(handle));

    checkCudaErr(cudaFree(dBuffer));
}