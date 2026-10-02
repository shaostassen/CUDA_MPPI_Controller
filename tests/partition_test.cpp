#include "mppi/partition.hpp"

#include <gtest/gtest.h>

#include <vector>

// The v2 reduction kernels (S2.4) decompose work two ways: one block per
// output element j -> (t, i), and threads striding over k within a block.
// Neither can be observed without a GPU, but both are pure index arithmetic,
// so the failure mode that matters — a slot covered twice or not at all —
// is checkable here. A GPU is only needed to measure the result, not to
// find out whether the partition is sound.

// j -> (t, i) must be a bijection onto the T x M grid.
TEST(Partition, OutputIndexSplitIsBijection) {
  for (int m : {1, 2, 3, 4}) {
    for (int T : {1, 5, 50}) {
      std::vector<int> hits(static_cast<std::size_t>(T) * m, 0);
      for (int j = 0; j < T * m; ++j) {
        int t = -1, i = -1;
        mppi::split_output_index(j, m, t, i);
        ASSERT_GE(t, 0);
        ASSERT_LT(t, T) << "j=" << j << " m=" << m;
        ASSERT_GE(i, 0);
        ASSERT_LT(i, m) << "j=" << j << " m=" << m;
        ++hits[static_cast<std::size_t>(t) * m + i];
      }
      for (std::size_t s = 0; s < hits.size(); ++s) {
        EXPECT_EQ(hits[s], 1) << "m=" << m << " T=" << T << " slot " << s;
      }
    }
  }
}

// The block-strided loop `for (k = tid; k < n; k += BlockSize)` must cover
// [0, n) exactly once across all threads — including when n is not a
// multiple of the block size, which is the case that breaks naive splits.
TEST(Partition, BlockStridedLoopCoversEveryItemOnce) {
  for (int block : {32, 128, 256}) {
    for (int n : {1, 31, 255, 256, 257, 1000, 4096}) {
      std::vector<int> hits(n, 0);
      int total = 0;
      for (int tid = 0; tid < block; ++tid) {
        int count = 0;
        for (int k = tid; k < n; k += block) {
          ++hits[k];
          ++count;
        }
        // strided_count must agree with the loop it describes.
        ASSERT_EQ(count, mppi::strided_count(tid, n, block))
            << "block=" << block << " n=" << n << " tid=" << tid;
        total += count;
      }
      EXPECT_EQ(total, n) << "block=" << block << " n=" << n;
      for (int k = 0; k < n; ++k) {
        EXPECT_EQ(hits[k], 1) << "block=" << block << " n=" << n << " item " << k;
      }
    }
  }
}

// Stage 1 of the two-stage reduction uses a grid-stride loop across blocks;
// the same cover property has to hold over the whole grid.
TEST(Partition, GridStrideLoopCoversEveryItemOnce) {
  const int block = 256;
  for (int grid : {1, 7, 64}) {
    for (int n : {1, 255, 4096, 100000}) {
      std::vector<int> hits(n, 0);
      for (int b = 0; b < grid; ++b) {
        for (int tid = 0; tid < block; ++tid) {
          for (int i = b * block + tid; i < n; i += block * grid) {
            ++hits[i];
          }
        }
      }
      for (int i = 0; i < n; ++i) {
        ASSERT_EQ(hits[i], 1) << "grid=" << grid << " n=" << n << " item " << i;
      }
    }
  }
}
