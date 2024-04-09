# SpMV-Samples

A collection of sample implementations for Sparse Matrix-Vector Multiplication (SpMV) using the Compressed Sparse Row (CSR) sparse matrix format on GPU.

Current Supported SpMV Implementations:
* [cuSparse SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv)
* [CUSP CSR-Vector SpMV](https://github.com/cusplibrary/cusplibrary/blob/main/cusp/system/cuda/detail/multiply/csr_vector_spmv.h) (supporting warp reduce)
* [LightSpMV](https://lightspmv.sourceforge.net/homepage.htm#latest)
* [Merge-based SpMV in CUB](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_spmv.cuh) (including a generalized SpMV implementation based on this method)

## Requirements
GCC (version 9.4.0 and above)  
CUDA (version 11.8 and above)  
CMake (version 3.12 and above)  

## Setup
Execute the following instructions in the root directory of the project.
```sh
mkdir build && cd build
cmake .. 
make
```

## Dataset
Support for **Matrix Market** format graph datasets.  
You can download graph datasets from the website https://sparse.tamu.edu/.

## Add Your SpMV
You can add your own CSR sparse matrix format SpMV implementation.  
Just encapsulate it with the following interface:
```cpp
// interface
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV(const std::string& kind_str,
    index_t n_rows,  index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y); 
```
And define the corresponding label strings and function definitions in the `spmv.h` header file.
```cpp
/// SPMV kind strings and its function
#define SPMV_KINDS                       \
    X("cusparse", SpMV_cusparse)         \
    X("YOUR_SPMV_LABEL", YOUR_SPMV_FUNC)
```