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
 * cub::AgentSegmentFixup implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
 */

#pragma once

#include <iterator>
#include <type_traits>

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>


namespace merge_genl {

// /// Optional outer namespace(s)
// CUB_NS_PREFIX

// /// CUB namespace
// namespace cub {

template <typename T, typename ReductionOpT>
__device__ 
typename std::enable_if_t<
    std::is_same_v<T, int> || std::is_same_v<T, unsigned>
>
AtomicReduce(T *address, T val, ReductionOpT reduce) {
   T old = *address, assumed;
   do {
      assumed = old;
      old = atomicCAS(address, assumed, reduce(val, assumed));
   } while (assumed != old);    
}

template <typename T, typename ReductionOpT>
__device__ 
typename std::enable_if_t<
    !std::is_same_v<T, int> && !std::is_same_v<T, unsigned> && 
    sizeof(T) == 4
>
AtomicReduce(T *address, T val, ReductionOpT reduce) {
    int * addr_as_int = reinterpret_cast<int *>(address);
    int old = *address, assumed{};
    T& assumed_T_ref = reinterpret_cast<T &>(assumed);
    T new_val{};
    int& new_val_int_ref = reinterpret_cast<int &>(new_val);
    do {
       assumed = old;
       new_val = reduce(val, assumed_T_ref);
       old = atomicCAS(addr_as_int, assumed, new_val_int_ref);
    } while (assumed != old);
}

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSegmentFixup
 */
template <
    int                             _BLOCK_THREADS,                 ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    cub::BlockLoadAlgorithm         _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    cub::CacheLoadModifier          _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    cub::BlockScanAlgorithm         _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct AgentSegmentFixupPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const cub::BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const cub::CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const cub::BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentSegmentFixup implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key
 */
template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    EqualityOpT,                    ///< KeyT equality operator type
    typename    ReductionOpT,                   ///< ValueT reduction operator type
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct AgentSegmentFixup
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of key-value input iterator
    typedef typename std::iterator_traits<PairsInputIteratorT>::value_type KeyValuePairT;

    using KeyT = typename KeyValuePairT::Key;

    // Value type
    typedef typename KeyValuePairT::Value ValueT;

    // Tile status descriptor interface type
    typedef cub::ReduceByKeyScanTileState<ValueT, OffsetT> ScanTileStateT;

    // Constants
    enum
    {
        BLOCK_THREADS       = AgentSegmentFixupPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentSegmentFixupPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not do fixup using RLE + global atomics
        USE_ATOMIC_FIXUP    = (CUB_PTX_ARCH >= 350) && 
                                (cub::Equals<ValueT, float>::VALUE || 
                                 cub::Equals<ValueT, int>::VALUE ||
                                 cub::Equals<ValueT, unsigned int>::VALUE ||
                                 cub::Equals<ValueT, unsigned long long>::VALUE),
        // USE_ATOMIC_FIXUP    = false,

        // Whether or not the scan operation has a zero-valued identity value (true if we're performing addition on a primitive type)
        HAS_IDENTITY_ZERO   = (cub::Equals<ReductionOpT, cub::Sum>::VALUE) && (cub::Traits<ValueT>::PRIMITIVE),
    };

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for keys
    typedef typename cub::If<cub::IsPointer<PairsInputIteratorT>::VALUE,
            cub::CacheModifiedInputIterator<AgentSegmentFixupPolicyT::LOAD_MODIFIER, KeyValuePairT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
        PairsInputIteratorT>::Type                                                                              // Directly use the supplied input iterator type
        WrappedPairsInputIteratorT;

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for fixup values
    typedef typename cub::If<cub::IsPointer<AggregatesOutputIteratorT>::VALUE,
            cub::CacheModifiedInputIterator<AgentSegmentFixupPolicyT::LOAD_MODIFIER, ValueT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
            AggregatesOutputIteratorT>::Type                                                             // Directly use the supplied input iterator type
        WrappedFixupInputIteratorT;

    // Reduce-value-by-segment scan operator
    typedef cub::ReduceByKeyOp<ReductionOpT> ReduceBySegmentOpT;

    // Parameterized BlockLoad type for pairs
    typedef cub::BlockLoad<
            KeyValuePairT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentSegmentFixupPolicyT::LOAD_ALGORITHM>
        BlockLoadPairs;

    // Parameterized BlockScan type
    typedef cub::BlockScan<
            KeyValuePairT,
            BLOCK_THREADS,
            AgentSegmentFixupPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef cub::TilePrefixCallbackOp<
            KeyValuePairT,
            ReduceBySegmentOpT,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Shared memory type for this threadblock
    union _TempStorage
    {
        struct ScanStorage
        {
            typename BlockScanT::TempStorage                scan;           // Smem needed for tile scanning
            typename TilePrefixCallbackOpT::TempStorage     prefix;         // Smem needed for cooperative prefix callback
        } scan_storage;

        // Smem needed for loading keys
        typename BlockLoadPairs::TempStorage load_pairs;
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : cub::Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&                        temp_storage;       ///< Reference to temp_storage
    WrappedPairsInputIteratorT           d_pairs_in;          ///< Input keys
    AggregatesOutputIteratorT            d_aggregates_out;   ///< Output value aggregates
    WrappedFixupInputIteratorT           d_fixup_in;         ///< Fixup input values
    cub::InequalityWrapper<EqualityOpT>  inequality_op;      ///< KeyT inequality operator
    ReductionOpT                         reduction_op;       ///< Reduction operator
    ReduceBySegmentOpT                   scan_op;            ///< Reduce-by-segment scan operator


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentSegmentFixup(
        TempStorage&                temp_storage,       ///< Reference to temp_storage
        PairsInputIteratorT         d_pairs_in,          ///< Input keys
        AggregatesOutputIteratorT   d_aggregates_out,   ///< Output value aggregates
        EqualityOpT                 equality_op,        ///< KeyT equality operator
        ReductionOpT                reduction_op)       ///< ValueT reduction operator
    :
        temp_storage(temp_storage.Alias()),
        d_pairs_in(d_pairs_in),
        d_aggregates_out(d_aggregates_out),
        d_fixup_in(d_aggregates_out),
        inequality_op(equality_op),
        reduction_op(reduction_op),
        scan_op(reduction_op)
    {}


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------


    /**
     * Process input tile.  Specialized for atomic-fixup
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT                 num_remaining,      ///< Number of global input items remaining (including this tile)
        int                     tile_idx,           ///< Tile index
        OffsetT                 tile_offset,        ///< Tile offset
        ScanTileStateT&         tile_state,         ///< Global tile state descriptor
        cub::Int2Type<true>     use_atomic_fixup)   ///< Marker whether to use atomicAdd (instead of reduce-by-key)
    {
        KeyValuePairT   pairs[ITEMS_PER_THREAD];

        // Load pairs
        KeyValuePairT oob_pair;
        oob_pair.key = static_cast<KeyT>(-1);      // 越界元素的默认值

        if (IS_LAST_TILE) {
            // 从全局内存d_pairs_in + tile_offset处加载num_remaining个数据,每个线程加载ITEMS_PER_THREAD个到pairs
            BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs, num_remaining, oob_pair);
        } else {
            // 从全局内存d_pairs_in + tile_offset处加载数据,每个线程加载ITEMS_PER_THREAD个到pairs
            BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs);
        }
        
        // RLE 
        #pragma unroll
        for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            // 输出内存的对应位置
            ValueT* d_scatter = d_aggregates_out + pairs[ITEM - 1].key;
            if (pairs[ITEM].key != pairs[ITEM - 1].key) {   // 与上一元素key不同,即对应矩阵的不同行
                // 将矩阵上一行的结果原子累加到输出
                AtomicReduce(d_scatter, pairs[ITEM - 1].value, reduction_op);
            } else {    // 与上一元素key相同,对应矩阵的同一行
                // 将上一元素值累加到当前元素
                pairs[ITEM].value = reduction_op(pairs[ITEM - 1].value, pairs[ITEM].value);
            }
        }

        // Flush last item if valid
        ValueT* d_scatter = d_aggregates_out + pairs[ITEMS_PER_THREAD - 1].key;
        // 不为最后一个线程块分片或者最后一个元素有key值(有效元素)
        if ((!IS_LAST_TILE) || (pairs[ITEMS_PER_THREAD - 1].key != static_cast<KeyT>(-1))) {
            // 写入输出内存
            AtomicReduce(d_scatter, pairs[ITEMS_PER_THREAD - 1].value, reduction_op);
        }
    }


    /**
     * Process input tile.  Specialized for reduce-by-key fixup
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT                 num_remaining,      ///< Number of global input items remaining (including this tile)
        int                     tile_idx,           ///< Tile index
        OffsetT                 tile_offset,        ///< Tile offset
        ScanTileStateT&         tile_state,         ///< Global tile state descriptor
        cub::Int2Type<false>    use_atomic_fixup)   ///< Marker whether to use atomicAdd (instead of reduce-by-key)
    {
        // 线程局部数据
        KeyValuePairT   pairs[ITEMS_PER_THREAD];
        // 前缀和数据
        KeyValuePairT   scatter_pairs[ITEMS_PER_THREAD];

        // Load pairs
        KeyValuePairT oob_pair;
        oob_pair.key = static_cast<KeyT>(-1);

        if (IS_LAST_TILE) {
            // 从全局内存d_pairs_in + tile_offset处加载num_remaining个数据,每个线程加载ITEMS_PER_THREAD个到pairs
            BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs, num_remaining, oob_pair);
        } else {
            // 从全局内存d_pairs_in + tile_offset处加载数据,每个线程加载ITEMS_PER_THREAD个到pairs
            BlockLoadPairs(temp_storage.load_pairs).Load(d_pairs_in + tile_offset, pairs);
        }

        __syncthreads();

        KeyValuePairT tile_aggregate;
        if (tile_idx == 0) {    // 全局(0,0)号线程块数据分片
            // Exclusive scan of values and segment_flags
            // 对线程局部数据求非包含前缀和
            BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(pairs, scatter_pairs, scan_op, tile_aggregate);

            // Update tile status if this is not the last tile
            if (threadIdx.x == 0) {
                // Set first segment id to not trigger a flush (invalid from exclusive scan)
                scatter_pairs[0].key = pairs[0].key;

                if (!IS_LAST_TILE) {
                    tile_state.SetInclusive(0, tile_aggregate);
                }
            }
        } else {    // 其他线程块数据分片
            // Exclusive scan of values and segment_flags
            // 前缀扫描的初始值运算符(用于计算每个线程块的起始偏移)
            TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_storage.prefix, scan_op, tile_idx);
            // 对线程局部数据求非包含前缀和
            BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(pairs, scatter_pairs, scan_op, prefix_op);
            // 线程块的前缀和
            tile_aggregate = prefix_op.GetBlockAggregate();  
        }

        // Scatter updated values
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            // 当前元素非包含前缀和的key与元素本身的key不同,即先前元素已归约完毕需写回
            if (scatter_pairs[ITEM].key != pairs[ITEM].key) {
                // Update the value at the key location
                // 读取原本输出位置的值(构造函数中d_fixup_in与d_aggregates_out相同)
                ValueT value    = d_fixup_in[scatter_pairs[ITEM].key];
                // 归约
                value           = reduction_op(value, scatter_pairs[ITEM].value);
                // 写回输出位置
                d_aggregates_out[scatter_pairs[ITEM].key] = value;
            }
        }

        // Finalize the last item
        if (IS_LAST_TILE) { // 最后一个线程块数据分片(num_remaining <= TILE_ITEMS)
            // Last thread will output final count and last item, if necessary
            if (threadIdx.x == BLOCK_THREADS - 1) { // 最后一个线程一定空闲
                // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
                // 仅当num_remaining == TILE_ITEMS时将最后值写回
                // num_remaining < TILE_ITEMS时上一个for循环已经写回
                if (num_remaining == TILE_ITEMS) {
                    // Update the value at the key location
                    OffsetT last_key = pairs[ITEMS_PER_THREAD - 1].key;
                    d_aggregates_out[last_key] = reduction_op(tile_aggregate.value, d_fixup_in[last_key]);
                }
            }
        }
    }


    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
        int                 num_tiles,          ///< Total number of input tiles
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index
        OffsetT tile_offset     = tile_idx * TILE_ITEMS;                    // Global offset for the current tile
        OffsetT num_remaining   = num_items - tile_offset;                  // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS) {
            // Not the last tile (full)
            // 不为最后一个线程块分片
            ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state, cub::Int2Type<USE_ATOMIC_FIXUP>());
        } else if (num_remaining > 0) {
            // The last tile (possibly partially-full)
            ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state, cub::Int2Type<USE_ATOMIC_FIXUP>());
        }
    }

};


// }               // CUB namespace
// CUB_NS_POSTFIX  // Optional outer namespace(s)

}   // namespace merge_genl