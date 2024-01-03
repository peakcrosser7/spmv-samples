/**
 * LightSpMV v1.0
 * 
 * Reference: https://ieeexplore.ieee.org/document/7245713
 * 			  https://lightspmv.sourceforge.net/homepage.htm#latest
 * 
*/

#pragma once

#include <cstdint>

#include "common.cuh"


#define MAX_NUM_THREADS_PER_BLOCK 1024

/*texture memory*/
#define USE_TEXTURE_MEMORY

/*error check*/
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__)
inline void __cudaCheckError(const char* file, const int32_t line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
            printf("cudaCheckError() failed at %s:%d : %s\n", file, line,
                   cudaGetErrorString(err));
            exit(-1);
	}
}

template <typename T>
cudaChannelFormatKind ChannelFormatKind = {};
template <>
cudaChannelFormatKind ChannelFormatKind<float> = cudaChannelFormatKindFloat;
template <>
cudaChannelFormatKind ChannelFormatKind<double> = cudaChannelFormatKindSigned;

inline void getKernelGridInfo(const int32_t dev,
        int32_t & numThreadsPerBlock, int32_t &numThreadBlocks) {

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, dev);
    /*set to the maximum number of threads per block*/
    numThreadsPerBlock = dev_prop.maxThreadsPerBlock;

    /*set to the number of multiprocessors*/
    numThreadBlocks =
        dev_prop.multiProcessorCount *
        (dev_prop.maxThreadsPerMultiProcessor / numThreadsPerBlock);

    //cerr << numThreadsPerBlock << " " << numThreadBlocks << endl;
}

/*device variables*/
__constant__ int32_t _cudaNumRows;
__constant__ int32_t _cudaNumCols;

/*macro to get the X value*/
template <typename T>
__device__ inline 
T VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index){}

template <>
__device__ inline 
float VECTOR_GET<float>(const cudaTextureObject_t vectorX, uint32_t index){
    return tex1Dfetch<float>(vectorX, index);
}

template <>
__device__ inline 
double VECTOR_GET<double>(const cudaTextureObject_t vectorX, uint32_t index){
	/*load the data*/
    // cannot get correct result
	// int2 v = tex1Dfetch<int2>(vectorX, index);

    int x = tex1Dfetch<int>(vectorX, index * 2);
    int y = tex1Dfetch<int>(vectorX, index * 2 + 1);

	/*convert to double*/
	return __hiloint2double(y, x);
}

template <typename T>
__device__ inline 
float VECTOR_GET(const T* __restrict vectorX, uint32_t index){
	return vectorX[index];
}

template <typename T>
__device__ __forceinline__
T shfl_down(T var, unsigned int delta, int width = 32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	return __shfl_down_sync(FULL_MASK, var, delta, width);
#else
	return __shfl_down(var, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__
T shfl(T var, int srcLane, int width = 32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	return __shfl_sync(FULL_MASK, var, srcLane, width);
#else
	return __shfl(var, srcLane, width);
#endif
}

/// vector-based row dynamic distribution
template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_t, typename vec_y_value_t>
__global__ void csrDynamicVector(
        index_t* __restrict cudaRowCounter,
        const offset_t* __restrict rowOffsets, const index_t* __restrict colIndexValues,
		const mat_value_t* __restrict numericalValues, 
        const vec_x_t vectorX,  vec_y_value_t* vectorY) {
	offset_t i;
	vec_y_value_t sum;
	index_t row;
	offset_t rowStart, rowEnd;
	const offset_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const offset_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the block*/
	__shared__ volatile offset_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (laneId == 0) {
		row = atomicAdd(cudaRowCounter, 1);
	}
	/*broadcast the value to other lanes from lane 0*/
	row = shfl(row, 0, THREADS_PER_VECTOR);

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}

            /*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;

			/*get a new row index*/
			row = atomicAdd(cudaRowCounter, 1);
		}
		row = shfl(row, 0, THREADS_PER_VECTOR);
	}/*while*/
}

/// warp-based row dynamic distribution
template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_t, typename vec_y_value_t>
__global__ void csrDynamicWarp(
        index_t* __restrict cudaRowCounter, 
        const offset_t* __restrict rowOffsets, const index_t* __restrict colIndexValues,
		const mat_value_t* __restrict numericalValues, 
        const vec_x_t vectorX,  vec_y_value_t* vectorY) {
	offset_t i;
	vec_y_value_t sum;
	index_t row;
	offset_t rowStart, rowEnd;
	const offset_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
	const offset_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
	const offset_t warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
	const offset_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;	/*vector index in the warp*/

	__shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

	/*get the row index*/
	if (warpLaneId == 0) {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	/*broadcast the value to other threads in the same warp and compute the row index of each vector*/
	row = shfl(row, 0) + warpVectorId;

	/*check the row range*/
	while (row < _cudaNumRows) {

		/*use two threads to fetch the row offset*/
		if (laneId < 2) {
			space[vectorId][laneId] = rowOffsets[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		/*there are non-zero elements in the current row*/
		sum = 0;
		/*compute dot product*/
		if (THREADS_PER_VECTOR == 32) {

			/*ensure aligned memory access*/
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			/*process the unaligned part*/
			if (i >= rowStart && i < rowEnd) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}

            /*process the aligned part*/
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}
		} else {
			/*regardless of the global memory access alignment*/
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
				sum += numericalValues[i] * VECTOR_GET<vec_y_value_t>(vectorX, colIndexValues[i]);
			}
		}
		/*intra-vector reduction*/
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
			sum += shfl_down(sum, i, THREADS_PER_VECTOR);
		}

		/*save the results and get a new row*/
		if (laneId == 0) {
			/*save the results*/
			vectorY[row] = sum;
		}

		/*get a new row index*/
		if(warpLaneId == 0){
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
		row = shfl(row, 0) + warpVectorId;

	}/*while*/
}

template <typename index_t, typename vec_x_value_t>
void LightSpMV_preprocess(
    index_t n_rows, index_t n_cols, const vec_x_value_t *x, 
    index_t*& _cudaRowCounters, cudaTextureObject_t& _texVectorX) {
    
    // set device cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    CudaCheckError();

    cudaMalloc(&_cudaRowCounters, sizeof(index_t));
    CudaCheckError();
    cudaMemset(_cudaRowCounters, 0, sizeof(index_t));
    CudaCheckError();

    cudaMemcpyToSymbol(_cudaNumRows, &n_rows, sizeof(index_t));
    CudaCheckError();

    cudaMemcpyToSymbol(_cudaNumCols, &n_cols, sizeof(index_t));
    CudaCheckError();

#ifdef USE_TEXTURE_MEMORY
	/*specify the texture object parameters*/
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;

    /*specify texture and texture object*/
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = const_cast<vec_x_value_t *>(x);
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
            ChannelFormatKind<vec_x_value_t>);
    resDesc.res.linear.sizeInBytes = n_cols * sizeof(vec_x_value_t);

    cudaCreateTextureObject(&_texVectorX, &resDesc, &texDesc, NULL);
    CudaCheckError();
#endif
}

template <typename index_t>
void LightSpMV_postprocess(index_t* _cudaRowCounters, cudaTextureObject_t& _texVectorX) {
#ifdef USE_TEXTURE_MEMORY
	cudaDestroyTextureObject(_texVectorX);
	CudaCheckError();
#endif
	cudaFree(_cudaRowCounters);
	CudaCheckError();
}

template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK,
          typename... args_t>
void csrDynamic(std::true_type, int32_t numThreadsPerBlock,
                int32_t numThreadBlocks, args_t &&...args) {
    csrDynamicVector<THREADS_PER_VECTOR, MAX_NUM_VECTORS_PER_BLOCK>
        <<<numThreadsPerBlock, numThreadBlocks>>>(
            std::forward<args_t>(args)...);
}

template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK,
          typename... args_t>
void csrDynamic(std::false_type, int32_t numThreadsPerBlock,
                int32_t numThreadBlocks, args_t &&...args) {
    csrDynamicWarp<THREADS_PER_VECTOR, MAX_NUM_VECTORS_PER_BLOCK>
        <<<numThreadsPerBlock, numThreadBlocks>>>(
            std::forward<args_t>(args)...);
}

template <typename USE_VECTOR,
          typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_t, typename vec_y_value_t>
void LightSpMV_invoke_kernel(
    index_t n_rows, offset_t nnz,
    index_t* _cudaRowCounters, 
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_t x, vec_y_value_t *y)  {

    int32_t _meanElementsPerRow = (int32_t)rint((double)nnz / n_rows);

    int32_t numThreadsPerBlock;
    int32_t numThreadBlocks;
    /*get the number of threads per block*/
    getKernelGridInfo(USED_DEVICE, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
    Timer::kernel_start();
    if (_meanElementsPerRow <= 2) {
        csrDynamic<2, MAX_NUM_THREADS_PER_BLOCK / 2>(
            USE_VECTOR{}, numThreadsPerBlock, numThreadBlocks, _cudaRowCounters,
            Ap, Aj, Ax, x, y);
    } else if (_meanElementsPerRow <= 4) {
        csrDynamic<4, MAX_NUM_THREADS_PER_BLOCK / 4>(
            USE_VECTOR{}, numThreadsPerBlock, numThreadBlocks, _cudaRowCounters,
            Ap, Aj, Ax, x, y);
    } else if (_meanElementsPerRow <= 64) {
        csrDynamic<8, MAX_NUM_THREADS_PER_BLOCK / 8>(
            USE_VECTOR{}, numThreadsPerBlock, numThreadBlocks, _cudaRowCounters,
            Ap, Aj, Ax, x, y);
    } else {
        csrDynamic<32, MAX_NUM_THREADS_PER_BLOCK / 32>(
            USE_VECTOR{}, numThreadsPerBlock, numThreadBlocks, _cudaRowCounters,
            Ap, Aj, Ax, x, y);
    }
    /*synchronize kernels*/
	cudaDeviceSynchronize();
    Timer::kernel_stop();
}

// LightSpMV Vector-level Dynamic Row Distribution
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_light_vector(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    index_t* _cudaRowCounters;
    cudaTextureObject_t _texVectorX;
    LightSpMV_preprocess(n_rows, n_cols, x, _cudaRowCounters, _texVectorX);

#ifdef USE_TEXTURE_MEMORY
    LightSpMV_invoke_kernel<std::true_type>(n_rows, nnz, _cudaRowCounters, Ap, Aj, Ax, _texVectorX, y);
#else
    LightSpMV_invoke_kernel<std::true_type>(n_rows, nnz, _cudaRowCounters, Ap, Aj, Ax, x, y);
#endif

	LightSpMV_postprocess(_cudaRowCounters, _texVectorX);
}

// LightSpMV Warp-level Dynamic Row Distribution
template <typename index_t, typename offset_t, typename mat_value_t,
          typename vec_x_value_t, typename vec_y_value_t>
void SpMV_light_warp(
    index_t n_rows, index_t n_cols, offset_t nnz,
    const offset_t *Ap, const index_t *Aj, const mat_value_t *Ax, 
    const vec_x_value_t *x, vec_y_value_t *y) {

    index_t* _cudaRowCounters;
    cudaTextureObject_t _texVectorX;
    LightSpMV_preprocess(n_rows, n_cols, x, _cudaRowCounters, _texVectorX);

#ifdef USE_TEXTURE_MEMORY
    LightSpMV_invoke_kernel<std::false_type>(n_rows, nnz, _cudaRowCounters, Ap, Aj, Ax, _texVectorX, y);
#else
    LightSpMV_invoke_kernel<std::false_type>(n_rows, nnz, _cudaRowCounters, Ap, Aj, Ax, x, y);
#endif

	LightSpMV_postprocess(_cudaRowCounters, _texVectorX);
}

