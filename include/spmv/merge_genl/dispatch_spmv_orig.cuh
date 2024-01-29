
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/util_type.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/grid/grid_queue.cuh>

#include "thread_search.cuh"
#include "agent_spmv_orig.cuh"
#include "agent_segment_fixup.cuh"

// /// Optional outer namespace(s)
// CUB_NS_PREFIX

/// CUB namespace
// namespace cub {

namespace merge_genl {

/******************************************************************************
 * SpMV kernel entry points
 *****************************************************************************/

/**
 * Spmv kernel for CSR with only one column.
 * 
 * 处理CSR只有1列(num_cols=1)的情况
 */
template <
    typename        AgentSpmvPolicyT,           ///< Parameterized SpmvPolicy tuning policy type
    typename        index_t,              ///< Matrix and vector value type
    typename        offset_t,             ///< Signed integer type for sequence offsets
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t,
    typename        functor_t>
__global__ void DeviceSpmv1ColKernel(
    SpmvParams<index_t, 
               offset_t, 
               mat_value_t, 
               vec_x_value_t,
               vec_y_value_t> spmv_params)                ///< [in] SpMV input parameter bundle
{
    using VectorValueIteratorT = cub::CacheModifiedInputIterator<
        AgentSpmvPolicyT::VECTOR_VALUES_LOAD_MODIFIER, 
        vec_x_value_t, 
        index_t>;
    using SpmvParamsT = SpmvParams<
        index_t, offset_t, 
        mat_value_t, vec_x_value_t, vec_y_value_t>;

    VectorValueIteratorT wrapped_vector_x(spmv_params.d_vector_x);

    // 每个线程负责一行直接计算SpMV
    index_t row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row_idx < spmv_params.num_rows) {
        offset_t     end_nonzero_idx = spmv_params.d_row_end_offsets[row_idx];
        offset_t     nonzero_idx = spmv_params.d_row_end_offsets[row_idx - 1];

        vec_y_value_t value = functor_t::initialize();
        if (end_nonzero_idx != nonzero_idx) {
            value = functor_t::reduce(value, 
                functor_t::combine(spmv_params.d_values[nonzero_idx], 
                        wrapped_vector_x[spmv_params.d_column_indices[nonzero_idx]]));
        }

        spmv_params.d_vector_y[row_idx] = value;
    }
}


/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 * 
 * 识别每个线程块分片的合并路径的起始坐标
 */
template <
    typename    SpmvPolicyT,                    ///< Parameterized SpmvPolicy tuning policy type
    typename    OffsetT,                        ///< Signed integer type for sequence offsets
    typename    CoordinateT,                    ///< Merge path coordinate type
    typename    SpmvParamsT>                    ///< SpmvParams type
__global__ void DeviceSpmvSearchKernel(
    int             num_merge_tiles,            ///< [in] Number of SpMV merge tiles (spmv grid size)
    CoordinateT*    d_tile_coordinates,         ///< [out] Pointer to the temporary array of tile starting coordinates
    SpmvParamsT     spmv_params)                ///< [in] SpMV input parameter bundle
{
    /// Constants
    enum
    {
        BLOCK_THREADS           = SpmvPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = SpmvPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    using RowOffsetsSearchIteratorT = cub::CacheModifiedInputIterator<
            SpmvPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    OffsetT tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_merge_tiles + 1) {
        OffsetT                              diagonal = (tile_idx * TILE_ITEMS);
        CoordinateT                          tile_coordinate;
        cub::CountingInputIterator<OffsetT>  nonzero_indices(0);

        // Search the merge path
        SearchMergePath(
            diagonal,
            RowOffsetsSearchIteratorT(spmv_params.d_row_end_offsets),   // 行偏移的迭代器
            nonzero_indices,    // 非零元索引的计数迭代器
            spmv_params.num_rows,
            spmv_params.num_nonzeros,
            tile_coordinate);   // 输出的起始坐标

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;
    }
}


/**
 * Spmv agent entry point
 */
template <
    typename        SpmvPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        index_t,
    typename        offset_t,
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t,
    typename        functor_t,
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        CoordinateT                 ///< Merge path coordinate type
>
__launch_bounds__ (int(SpmvPolicyT::BLOCK_THREADS))
__global__ void DeviceSpmvKernel(
    SpmvParams<index_t, 
               offset_t, 
               mat_value_t, 
               vec_x_value_t,
               vec_y_value_t>                       spmv_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                                d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    cub::KeyValuePair<offset_t, vec_y_value_t>* d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                                         num_tiles,                  ///< [in] Number of merge tiles
    ScanTileStateT                              tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                                         num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
{
    // Spmv agent type specialization
    typedef AgentSpmv<SpmvPolicyT,
            index_t, offset_t, 
            mat_value_t, vec_x_value_t, vec_y_value_t, 
            functor_t>
        AgentSpmvT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentSpmvT::TempStorage temp_storage;

    AgentSpmvT(temp_storage, spmv_params).ConsumeTile(
        d_tile_coordinates,
        d_tile_carry_pairs,     // 该线程块分片的最后一行的行偏移和累加结果
        num_tiles);

    // Initialize fixup tile status
    tile_state.InitializeStatus(num_segment_fixup_tiles);

}

/**
 * Multi-block reduce-by-key sweep kernel entry point
 */
template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    KeyValuePairT,                  ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    offset_t,                        ///< Signed integer type for global offsets
    typename    ScanTileStateT,                 ///< Tile status interface type
    typename    functor_t>
__launch_bounds__ (int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
__global__ void DeviceSegmentFixupKernel(
    KeyValuePairT*              d_pairs_in,         ///< [in] Pointer to the array carry-out dot product row-ids, one per spmv block
    AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    offset_t                    num_items,          ///< [in] Total number of items to select from
    offset_t                    num_tiles,          ///< [in] Total number of tiles for the entire problem
    ScanTileStateT              tile_state)         ///< [in] Tile status interface
{
    using ReduceT = ReduceWrapper<functor_t, typename KeyValuePairT::Value>;

    // Thread block type for reducing tiles of value segments
    using AgentSegmentFixupT = AgentSegmentFixup<
            AgentSegmentFixupPolicyT,
            KeyValuePairT*,
            AggregatesOutputIteratorT,
            cub::Equality,
            ReduceT,
            offset_t>;

    // Shared memory for AgentSegmentFixup
    __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;

    // Process tiles
    AgentSegmentFixupT(temp_storage, d_pairs_in, d_aggregates_out, cub::Equality(), ReduceT()).ConsumeRange(
        num_items,
        num_tiles,
        tile_state);
}


/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
 */
template <
    typename        index_t,              ///< Matrix and vector value type
    typename        offset_t,             ///< Signed integer type for sequence offsets
    typename        mat_value_t,
    typename        vec_x_value_t,
    typename        vec_y_value_t,
    typename        functor_t>
struct DispatchSpmv {
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum {
        INIT_KERNEL_THREADS = 128
    };

    // SpmvParams bundle type
    typedef SpmvParams<index_t, 
                       offset_t, 
                       mat_value_t, 
                       vec_x_value_t,
                       vec_y_value_t> SpmvParamsT;

    // 2D merge path coordinate type
    typedef typename cub::CubVector<offset_t, 2>::Type CoordinateT;

    // Tile status descriptor interface type
    typedef cub::ReduceByKeyScanTileState<vec_y_value_t, offset_t> ScanTileStateT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef cub::KeyValuePair<offset_t, vec_y_value_t> KeyValuePairT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------

    constexpr static size_t max_value_byte = std::max(std::max(sizeof(mat_value_t), sizeof(vec_x_value_t)),
                                                      sizeof(vec_y_value_t));

    /// SM11
    struct Policy110 {
        using SpmvPolicyT = AgentSpmvPolicy<
                128,
                1,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                false,
                cub::BLOCK_SCAN_WARP_SCANS>;

        typedef AgentSegmentFixupPolicy<
                128,
                4,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };

    /// SM20
    struct Policy200 {
        using SpmvPolicyT = AgentSpmvPolicy<
                96,
                18,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                false,
                cub::BLOCK_SCAN_RAKING>;

        typedef AgentSegmentFixupPolicy<
                128,
                4,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;

    };



    /// SM30
    struct Policy300 {
        using SpmvPolicyT = AgentSpmvPolicy<
                96,
                6,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                false,
                cub::BLOCK_SCAN_WARP_SCANS>;

        typedef AgentSegmentFixupPolicy<
                128,
                4,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_DEFAULT,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;

    };


    /// SM35
    struct Policy350 {
        using SpmvPolicyT = AgentSpmvPolicy<
                (max_value_byte > 4) ? 96 : 128,
                (max_value_byte > 4) ? 4 : 7,
                cub::LOAD_LDG,
                cub::LOAD_CA,
                cub::LOAD_LDG,
                cub::LOAD_LDG,
                cub::LOAD_LDG,
                (max_value_byte > 4) ? true : false,
                cub::BLOCK_SCAN_WARP_SCANS>;

        typedef AgentSegmentFixupPolicy<
                128,
                3,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_LDG,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };


    /// SM37
    struct Policy370 {
        using SpmvPolicyT = AgentSpmvPolicy<
                (max_value_byte > 4) ? 128 : 128,
                (max_value_byte > 4) ? 9 : 14,
                cub::LOAD_LDG,
                cub::LOAD_CA,
                cub::LOAD_LDG,
                cub::LOAD_LDG,
                cub::LOAD_LDG,
                false, 
                cub::BLOCK_SCAN_WARP_SCANS>;

        typedef AgentSegmentFixupPolicy<
                128,
                3,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_LDG,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };

    /// SM50
    struct Policy500 {
        using SpmvPolicyT = AgentSpmvPolicy<
                (max_value_byte > 4) ? 64 : 128,
                (max_value_byte > 4) ? 6 : 7,
                cub::LOAD_LDG,
                cub::LOAD_DEFAULT,
                (max_value_byte > 4) ? cub::LOAD_LDG : cub::LOAD_DEFAULT,
                (max_value_byte > 4) ? cub::LOAD_LDG : cub::LOAD_DEFAULT,
                cub::LOAD_LDG,
                (max_value_byte > 4) ? true : false,
                (max_value_byte > 4) ? cub::BLOCK_SCAN_WARP_SCANS : cub::BLOCK_SCAN_RAKING_MEMOIZE>;

        typedef AgentSegmentFixupPolicy<
                128,
                3,
                cub::BLOCK_LOAD_VECTORIZE,
                cub::LOAD_LDG,
                cub::BLOCK_SCAN_RAKING_MEMOIZE>
            SegmentFixupPolicyT;
    };

    /// SM60
    struct Policy600 {
        using SpmvPolicyT = AgentSpmvPolicy<
                (max_value_byte > 4) ? 64 : 128,
                (max_value_byte > 4) ? 5 : 7,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                cub::LOAD_DEFAULT,
                false,
                cub::BLOCK_SCAN_WARP_SCANS>;

        typedef AgentSegmentFixupPolicy<
                128,
                3,
                cub::BLOCK_LOAD_DIRECT,
                cub::LOAD_LDG,
                cub::BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;
    };


    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

#if (CUB_PTX_ARCH >= 600)
    typedef Policy600 PtxPolicy;
#elif (CUB_PTX_ARCH >= 500)
    typedef Policy500 PtxPolicy;
#elif (CUB_PTX_ARCH >= 370)
    typedef Policy370 PtxPolicy;
#elif (CUB_PTX_ARCH >= 350)
    typedef Policy350 PtxPolicy;
#elif (CUB_PTX_ARCH >= 300)
    typedef Policy300 PtxPolicy;
#elif (CUB_PTX_ARCH >= 200)
    typedef Policy200 PtxPolicy;
#else
    typedef Policy110 PtxPolicy;
#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSpmvPolicyT : PtxPolicy::SpmvPolicyT {};
    struct PtxSegmentFixupPolicy : PtxPolicy::SegmentFixupPolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &spmv_config,
        KernelConfig    &segment_fixup_config) {
    #if (CUB_PTX_ARCH > 0)

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        spmv_config.template Init<PtxSpmvPolicyT>();
        segment_fixup_config.template Init<PtxSegmentFixupPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 600) {
            spmv_config.template            Init<typename Policy600::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy600::SegmentFixupPolicyT>();
        } else if (ptx_version >= 500) {
            spmv_config.template            Init<typename Policy500::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy500::SegmentFixupPolicyT>();
        } else if (ptx_version >= 370) {
            spmv_config.template            Init<typename Policy370::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy370::SegmentFixupPolicyT>();
        } else if (ptx_version >= 350) {
            spmv_config.template            Init<typename Policy350::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy350::SegmentFixupPolicyT>();
        } else if (ptx_version >= 300) {
            spmv_config.template            Init<typename Policy300::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy300::SegmentFixupPolicyT>();
        } else if (ptx_version >= 200) {
            spmv_config.template            Init<typename Policy200::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy200::SegmentFixupPolicyT>();
        } else {
            spmv_config.template            Init<typename Policy110::SpmvPolicyT>();
            segment_fixup_config.template   Init<typename Policy110::SegmentFixupPolicyT>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig {
        int block_threads;
        int items_per_thread;
        /// @brief 线程块需处理的元素总数
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init() {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        typename                Spmv1ColKernelT,                    ///< Function type of cub::DeviceSpmv1ColKernel
        typename                SpmvSearchKernelT,                  ///< Function type of cub::AgentSpmvSearchKernel
        typename                SpmvKernelT,                        ///< Function type of cub::AgentSpmvKernel
        typename                SegmentFixupKernelT>                 ///< Function type of cub::DeviceSegmentFixupKernelT
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,                  ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        Spmv1ColKernelT         spmv_1col_kernel,                   ///< [in] Kernel function pointer to parameterization of DeviceSpmv1ColKernel
        SpmvSearchKernelT       spmv_search_kernel,                 ///< [in] Kernel function pointer to parameterization of AgentSpmvSearchKernel
        SpmvKernelT             spmv_kernel,                        ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        SegmentFixupKernelT     segment_fixup_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceSegmentFixupKernel
        KernelConfig            spmv_config,                        ///< [in] Dispatch parameters that match the policy that \p spmv_kernel was compiled for
        KernelConfig            segment_fixup_config)               ///< [in] Dispatch parameters that match the policy that \p segment_fixup_kernel was compiled for
    {
        cudaError error = cudaSuccess;
        do {
            if constexpr (std::is_signed_v<index_t>) {
                if (spmv_params.num_rows < 0 || spmv_params.num_cols < 0) {
                    return cudaErrorInvalidValue;
                }
            }

            if (spmv_params.num_rows == 0 || spmv_params.num_cols == 0) { // Empty problem, no-op.
                if (d_temp_storage == NULL) {
                    temp_storage_bytes = 1;
                }

                break;
            }

            if (spmv_params.num_cols == 1) {
                if (d_temp_storage == NULL) {
                    // Return if the caller is simply requesting the size of the storage allocation
                    temp_storage_bytes = 1;
                    break;
                }

                // Get search/init grid dims
                int degen_col_kernel_block_size     = INIT_KERNEL_THREADS;
                int degen_col_kernel_grid_size      = cub::DivideAndRoundUp(spmv_params.num_rows, degen_col_kernel_block_size);

                if (debug_synchronous) _CubLog("Invoking spmv_1col_kernel<<<%d, %d, 0, %lld>>>()\n",
                    degen_col_kernel_grid_size, degen_col_kernel_block_size, (long long) stream);

                // Invoke spmv_kernel
                spmv_1col_kernel<<<degen_col_kernel_grid_size, degen_col_kernel_block_size, 0, stream>>>(
                    spmv_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;

                break;
            }

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;;

            // Total number of spmv work items
            // SpMV总共需要需要合并的元素个数
            offset_t num_merge_items = offset_t(spmv_params.num_rows) + spmv_params.num_nonzeros;

            // Tile sizes of kernels
            // SpMV kernel线程块总共需要处理的元素数
            int merge_tile_size              = spmv_config.block_threads * spmv_config.items_per_thread;
            // fixup kernel线程块总共需要处理的元素数
            int segment_fixup_tile_size     = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;

            // Number of tiles for kernels
            // SpMV kernel所需的线程块数
            offset_t num_merge_tiles            = cub::DivideAndRoundUp(num_merge_items, merge_tile_size);
            // fixup kernel所需的线程块个数
            offset_t num_segment_fixup_tiles    = cub::DivideAndRoundUp(num_merge_tiles, segment_fixup_tile_size);

            // Get SM occupancy for kernels
            // 每个SM可驻留的SpMV kernel的线程块的最大数
            int spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                spmv_sm_occupancy,
                spmv_kernel,
                spmv_config.block_threads))) break;

            // 每个SM可驻留的fixup kernel的线程块的最大数
            int segment_fixup_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                segment_fixup_sm_occupancy,
                segment_fixup_kernel,
                segment_fixup_config.block_threads))) break;

            // Get grid dimensions
            dim3 spmv_grid_size(
                CUB_MIN(num_merge_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_merge_tiles, max_dim_x),
                1);

            dim3 segment_fixup_grid_size(
                CUB_MIN(num_segment_fixup_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_segment_fixup_tiles, max_dim_x),
                1);

            // Get the temporary storage allocation requirements
            size_t allocation_sizes[3]; // 3表示需要3部分临时内存
            // bytes needed for reduce-by-key tile status descriptors
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[0]))) break;
            // bytes needed for block carry-out pairs
            allocation_sizes[1] = num_merge_tiles * sizeof(KeyValuePairT);
            // bytes needed for tile starting coordinates
            allocation_sizes[2] = (num_merge_tiles + 1) * sizeof(CoordinateT);

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            void* allocations[3] = {};
            // 计算256字节对齐后所需临时内存大小,并将对应部分的内存指针存到allocations数组
            if (CubDebug(error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL) {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Construct the tile status interface
            ScanTileStateT tile_state;
            // 设置用于scan操作的设备内存
            if (CubDebug(error = tile_state.Init(num_segment_fixup_tiles, allocations[0], allocation_sizes[0]))) break;

            // Alias the other allocations
            KeyValuePairT*  d_tile_carry_pairs      = (KeyValuePairT*) allocations[1];  // Agent carry-out pairs
            CoordinateT*    d_tile_coordinates      = (CoordinateT*) allocations[2];    // Agent starting coordinates

            // Get search/init grid dims
            int search_block_size   = INIT_KERNEL_THREADS;
            // 前缀和需要总数+1
            offset_t search_grid_size    = cub::DivideAndRoundUp(num_merge_tiles + 1, search_block_size);


            if (search_grid_size < sm_count) {
                // Not enough spmv tiles to saturate the device: have spmv blocks search their own staring coords
                // search kernel的网格数少于SM数,则让不使用单独的search kernel搜索起始坐标,而在SpMV kernel中搜索
                d_tile_coordinates = NULL;
            } else {
                // Use separate search kernel if we have enough spmv tiles to saturate the device

                // Log spmv_search_kernel configuration
                if (debug_synchronous) _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                    search_grid_size, search_block_size, (long long) stream);

                // Invoke spmv_search_kernel
                // 计算每个线程块分片的合并序列的起始坐标
                spmv_search_kernel<<<search_grid_size, search_block_size, 0, stream>>>(
                    num_merge_tiles,
                    d_tile_coordinates,
                    spmv_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;
            }

            // Log spmv_kernel configuration
            if (debug_synchronous) _CubLog("Invoking spmv_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                spmv_grid_size.x, spmv_grid_size.y, spmv_grid_size.z, spmv_config.block_threads, (long long) stream, spmv_config.items_per_thread, spmv_sm_occupancy);

            // Invoke spmv_kernel
            spmv_kernel<<<spmv_grid_size, spmv_config.block_threads, 0, stream>>>(
                spmv_params,
                d_tile_coordinates,
                d_tile_carry_pairs,
                num_merge_tiles,
                tile_state,
                num_segment_fixup_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;

            // Run reduce-by-key fixup if necessary
            // 线程块分片大于1时需要修复跨行结果
            if (num_merge_tiles > 1) {
                // Log segment_fixup_kernel configuration
                if (debug_synchronous) _CubLog("Invoking segment_fixup_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    segment_fixup_grid_size.x, segment_fixup_grid_size.y, segment_fixup_grid_size.z, segment_fixup_config.block_threads, (long long) stream, segment_fixup_config.items_per_thread, segment_fixup_sm_occupancy);

                // Invoke segment_fixup_kernel
                segment_fixup_kernel<<<segment_fixup_grid_size, segment_fixup_config.block_threads, 0, stream>>>(
                    d_tile_carry_pairs,
                    spmv_params.d_vector_y,
                    num_merge_tiles,
                    num_segment_fixup_tiles,
                    tile_state);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;
            }
        } while (0);

        return error;
    }


    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        SpmvParamsT&            spmv_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream                  = 0,        ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous       = false)    ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do {
            // Get PTX version
            int ptx_version;
    
            if (CubDebug(error = cub::PtxVersion(ptx_version))) break;


            // Get kernel kernel dispatch configurations
            KernelConfig spmv_config, segment_fixup_config;
            InitConfigs(ptx_version, spmv_config, segment_fixup_config);

            if (CubDebug(error = Dispatch(
                d_temp_storage, temp_storage_bytes, spmv_params, stream, debug_synchronous,
                DeviceSpmv1ColKernel<PtxSpmvPolicyT, index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t, functor_t>,
                DeviceSpmvSearchKernel<PtxSpmvPolicyT, offset_t, CoordinateT, SpmvParamsT>,
                DeviceSpmvKernel<PtxSpmvPolicyT, index_t, offset_t, mat_value_t, vec_x_value_t, vec_y_value_t, functor_t, ScanTileStateT, CoordinateT>,
                DeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT, vec_y_value_t*, offset_t, ScanTileStateT, functor_t>,
                spmv_config, segment_fixup_config))) break;
        } while (0);

        return error;
    }
};

}   // namespace merge_genl

// }               // CUB namespace
// CUB_NS_POSTFIX  // Optional outer namespace(s)


