#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "mppi/philox.hpp"

using mppi::rng::normal_pair;
using mppi::rng::Philox4x32_10;
using mppi::rng::u01;

// S1.2 AC: known-answer tests. Vectors are from Random123's tests/kat_vectors
// (DEShawResearch/random123), the reference implementation by the Philox
// authors — not derived from this codebase.

TEST(Philox, KnownAnswerZeroInput) {
  auto r = Philox4x32_10::round10({{0, 0, 0, 0}}, {{0, 0}});
  EXPECT_EQ(r.w[0], 0x6627e8d5u);
  EXPECT_EQ(r.w[1], 0xe169c58du);
  EXPECT_EQ(r.w[2], 0xbc57ac4cu);
  EXPECT_EQ(r.w[3], 0x9b00dbd8u);
}

TEST(Philox, KnownAnswerAllOnesInput) {
  auto r = Philox4x32_10::round10({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}},
                                  {{0xffffffffu, 0xffffffffu}});
  EXPECT_EQ(r.w[0], 0x408f276du);
  EXPECT_EQ(r.w[1], 0x41c83b0eu);
  EXPECT_EQ(r.w[2], 0xa20bc7c6u);
  EXPECT_EQ(r.w[3], 0x6d5451fdu);
}

TEST(Philox, KnownAnswerPiDigitsInput) {
  auto r = Philox4x32_10::round10({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}},
                                  {{0xa4093822u, 0x299f31d0u}});
  EXPECT_EQ(r.w[0], 0xd16cfe09u);
  EXPECT_EQ(r.w[1], 0x94fdccebu);
  EXPECT_EQ(r.w[2], 0x5001e420u);
  EXPECT_EQ(r.w[3], 0x24126ea1u);
}

// block() is round10 with the pinned (seed, iteration, k, t) counter layout
// and a zero key, so the zero KAT must hold through it too.
TEST(Philox, BlockHelperMatchesKATAtZero) {
  auto r = Philox4x32_10::block(0, 0, 0, 0);
  EXPECT_EQ(r.w[0], 0x6627e8d5u);
  EXPECT_EQ(r.w[1], 0xe169c58du);
  EXPECT_EQ(r.w[2], 0xbc57ac4cu);
  EXPECT_EQ(r.w[3], 0x9b00dbd8u);
}

// Catches plumbing bugs where a counter field isn't actually wired in.
TEST(Philox, EveryCounterFieldChangesOutput) {
  auto base = Philox4x32_10::block(1, 2, 3, 4);
  auto d_seed = Philox4x32_10::block(2, 2, 3, 4);
  auto d_iter = Philox4x32_10::block(1, 3, 3, 4);
  auto d_k = Philox4x32_10::block(1, 2, 4, 4);
  auto d_t = Philox4x32_10::block(1, 2, 3, 5);
  for (auto* other : {&d_seed, &d_iter, &d_k, &d_t}) {
    bool any_diff = false;
    for (int i = 0; i < 4; ++i) {
      any_diff = any_diff || (other->w[i] != base.w[i]);
    }
    EXPECT_TRUE(any_diff);
  }
}

TEST(PhiloxU01, StaysInHalfOpenUnitInterval) {
  EXPECT_GT(u01(0u), 0.0f);
  EXPECT_LE(u01(0xffffffffu), 1.0f);
  for (std::uint32_t k = 0; k < 1000; ++k) {
    auto b = Philox4x32_10::block(7, 0, k, 0);
    for (int i = 0; i < 4; ++i) {
      float u = u01(b.w[i]);
      EXPECT_GT(u, 0.0f);
      EXPECT_LE(u, 1.0f);
    }
  }
}

// Sanity, not part of the AC: fixed-seed sample moments should look N(0,1).
// Deterministic (same counters every run), so this cannot flake.
TEST(PhiloxNormals, SampleMomentsLookStandardNormal) {
  constexpr int kBlocks = 1 << 17;  // 4 normals per block
  double sum = 0.0, sum_sq = 0.0;
  for (std::uint32_t i = 0; i < kBlocks; ++i) {
    auto b = Philox4x32_10::block(42, 0, i, 0);
    float z0, z1, z2, z3;
    normal_pair(b.w[0], b.w[1], z0, z1);
    normal_pair(b.w[2], b.w[3], z2, z3);
    for (float z : {z0, z1, z2, z3}) {
      ASSERT_TRUE(std::isfinite(z));
      sum += z;
      sum_sq += static_cast<double>(z) * z;
    }
  }
  double n = 4.0 * kBlocks;
  double mean = sum / n;
  double var = sum_sq / n - mean * mean;
  EXPECT_NEAR(mean, 0.0, 0.01);
  EXPECT_NEAR(var, 1.0, 0.02);
}
