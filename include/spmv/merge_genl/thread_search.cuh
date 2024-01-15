#pragma once

#include <iterator>

namespace merge_genl {

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
__host__ __device__ __forceinline__ 
void SearchMergePath(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate) {

    /// The value type of the input iterator
    typedef typename std::iterator_traits<AIteratorT>::value_type T;

    // 二分搜索最小索引,初始化为对角线的a方向最小索引
    OffsetT split_min = CUB_MAX(diagonal - b_len, 0);
    // 二分搜索最大索引,初始化为对角线的a方向最大索引
    OffsetT split_max = CUB_MIN(diagonal, a_len);

    // 二分搜索
    // i+j=k,寻找第一个满足A_i>B_j的坐标(i,j)
    // 在这里,split_pivot即i,diagonal即k
    while (split_min < split_max) {
        OffsetT split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1]) {
            // Move candidate split range up A, down B
            split_min = split_pivot + 1;
        } else {
            // Move candidate split range up B, down A
            split_max = split_pivot;
        }
    }

    // 确保坐标在0~a_len的有效范围(可能多余)
    path_coordinate.x = CUB_MIN(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}

}   // namespace merge_genl