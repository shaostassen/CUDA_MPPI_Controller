#pragma once

#include <math.h>

#include <cstdint>

#include "mppi/macros.hpp"

namespace mppi::rng {

// Counter-based Philox4x32-10 (Salmon et al., SC'11), one integer-only code
// path shared by host and device so raw outputs are bit-identical across
// CPU and GPU — that is what makes the S2.x parity tests strict.
// Counter layout is pinned in PLAN.md §1 decision 4: (seed, iteration,
// rollout k, timestep t); one block yields 4 words = up to 4 control dims
// per (k, t), which the quadrotor's m = 4 consumes exactly.
struct Philox4x32_10 {
  struct Block {
    std::uint32_t w[4];
  };
  struct Key {
    std::uint32_t k[2];
  };

  MPPI_HOST_DEVICE static Block round10(Block ctr, Key key) {
    ctr = one_round(ctr, key);
    for (int i = 0; i < 9; ++i) {
      key.k[0] += kWeyl0;
      key.k[1] += kWeyl1;
      ctr = one_round(ctr, key);
    }
    return ctr;
  }

  MPPI_HOST_DEVICE static Block block(std::uint32_t seed, std::uint32_t iteration,
                                      std::uint32_t rollout, std::uint32_t timestep) {
    return round10(Block{{seed, iteration, rollout, timestep}}, Key{{0, 0}});
  }

 private:
  static constexpr std::uint32_t kMult0 = 0xD2511F53u;
  static constexpr std::uint32_t kMult1 = 0xCD9E8D57u;
  static constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
  static constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;

  // Plain uint64 multiply rather than __umulhi so host and device share one
  // code path; revisit only if ncu ever shows this multiply mattering.
  MPPI_HOST_DEVICE static void mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi,
                                       std::uint32_t& lo) {
    std::uint64_t p = static_cast<std::uint64_t>(a) * b;
    hi = static_cast<std::uint32_t>(p >> 32);
    lo = static_cast<std::uint32_t>(p);
  }

  MPPI_HOST_DEVICE static Block one_round(const Block& c, const Key& k) {
    std::uint32_t hi0, lo0, hi1, lo1;
    mulhilo(kMult0, c.w[0], hi0, lo0);
    mulhilo(kMult1, c.w[2], hi1, lo1);
    return Block{{hi1 ^ c.w[1] ^ k.k[0], lo1, hi0 ^ c.w[3] ^ k.k[1], lo0}};
  }
};

// Maps a word to (0, 1] with 24-bit resolution. Exact arithmetic (one
// int->float convert + one multiply), so u01 is also bit-identical host vs
// device.
MPPI_HOST_DEVICE inline float u01(std::uint32_t x) {
  return (static_cast<float>(x >> 8) + 1.0f) * 0x1.0p-24f;
}

// Box-Muller. logf/sinf/cosf are the one place host/device bit-parity can
// slip (device transcendentals sit a few ULP off libm), so cross CPU<->GPU
// comparisons of normals are tolerance-based; raw blocks and u01 stay
// bit-exact. u1 >= 2^-24 bounds r at ~5.8 sigma, so no infinities ever.
MPPI_HOST_DEVICE inline void normal_pair(std::uint32_t a, std::uint32_t b, float& z0, float& z1) {
  float u1 = u01(a);
  float u2 = u01(b);
  float r = sqrtf(-2.0f * logf(u1));
  float theta = 6.283185307179586f * u2;
  z0 = r * cosf(theta);
  z1 = r * sinf(theta);
}

}  // namespace mppi::rng
