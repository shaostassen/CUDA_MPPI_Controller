#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "bench/config.hpp"
#include "bench/power.hpp"
#include "bench/stats.hpp"

// The harness itself has to be trustworthy before any number it prints
// means anything. Statistics, config parsing, and tegrastats parsing are all
// plain host code, so they get tested like anything else — a p99 that is
// silently a p95, or a power column that misreads a rail, would corrupt
// Table 1 without ever looking wrong.

TEST(BenchStats, PercentilesUseNearestRank) {
  std::vector<double> v(100);
  for (int i = 0; i < 100; ++i) {
    v[i] = i + 1;  // 1..100
  }
  auto s = bench::summarize(v);
  EXPECT_DOUBLE_EQ(s.min_ms, 1.0);
  EXPECT_DOUBLE_EQ(s.max_ms, 100.0);
  EXPECT_DOUBLE_EQ(s.mean_ms, 50.5);
  EXPECT_DOUBLE_EQ(s.p50_ms, 50.0);
  EXPECT_DOUBLE_EQ(s.p99_ms, 99.0);
  EXPECT_EQ(s.count, 100u);
}

// The tail is the point of reporting p99 at all: it must track the outliers,
// not be averaged away by the other 490 samples.
TEST(BenchStats, P99TracksTheTailNotTheMean) {
  std::vector<double> v(500, 1.0);
  for (int i = 0; i < 10; ++i) {
    v[i] = 40.0;  // 2% of samples spike, so the top 1% is entirely spikes
  }
  auto s = bench::summarize(v);
  EXPECT_NEAR(s.mean_ms, 1.78, 0.01);
  EXPECT_DOUBLE_EQ(s.p50_ms, 1.0);
  EXPECT_DOUBLE_EQ(s.p99_ms, 40.0) << "p99 must expose the spike a deadline would miss";
}

// At exactly 1% spikes the nearest-rank p99 is the last sample below them —
// 99% of observations really are 1.0. Pinned because it looks like an
// off-by-one when read quickly, and someone will eventually "fix" it.
TEST(BenchStats, P99BoundaryAtExactlyOnePercent) {
  std::vector<double> v(500, 1.0);
  for (int i = 0; i < 5; ++i) {
    v[i] = 40.0;
  }
  auto s = bench::summarize(v);
  EXPECT_DOUBLE_EQ(s.p99_ms, 1.0);
  EXPECT_DOUBLE_EQ(s.max_ms, 40.0) << "the spike is still visible in max";
}

TEST(BenchStats, HandlesEmptyAndSingleSample) {
  auto empty = bench::summarize({});
  EXPECT_EQ(empty.count, 0u);
  EXPECT_DOUBLE_EQ(empty.p99_ms, 0.0);

  auto one = bench::summarize({7.5});
  EXPECT_EQ(one.count, 1u);
  EXPECT_DOUBLE_EQ(one.p50_ms, 7.5);
  EXPECT_DOUBLE_EQ(one.p99_ms, 7.5);
}

TEST(BenchConfig, ParsesAndAppliesDefaults) {
  auto c = bench::parse_config(R"({"name":"t","samples":2048,"horizon":25})");
  EXPECT_EQ(c.name, "t");
  EXPECT_EQ(c.samples, 2048);
  EXPECT_EQ(c.horizon, 25);
  EXPECT_EQ(c.warmup, 50);   // §4 protocol defaults
  EXPECT_EQ(c.measure, 500);
  EXPECT_DOUBLE_EQ(c.dt, 0.02);
}

TEST(BenchConfig, ReadsBackendList) {
  auto c = bench::parse_config(R"({"name":"t","backends":["cpu_1t","cuda_v2"]})");
  ASSERT_EQ(c.backends.size(), 2u);
  EXPECT_EQ(c.backends[0], "cpu_1t");
  EXPECT_EQ(c.backends[1], "cuda_v2");
}

// A malformed config must fail loudly at load, not produce a plausible row.
TEST(BenchConfig, RejectsInvalidValues) {
  EXPECT_THROW(bench::parse_config(R"({"name":"t","samples":0})"), std::exception);
  EXPECT_THROW(bench::parse_config(R"({"name":"t","horizon":-5})"), std::exception);
  EXPECT_THROW(bench::parse_config(R"({"name":"t","sigma":0})"), std::exception);
  EXPECT_THROW(bench::parse_config(R"({"name":"t","measure":0})"), std::exception);
  EXPECT_THROW(bench::parse_config("{not json"), std::exception);
}

// A real tegrastats line from a JetPack 6 Orin, abbreviated in the middle.
constexpr const char* kTegraLine =
    "09-15-2026 10:23:45 RAM 3456/7620MB (lfb 12x4MB) SWAP 0/3810MB (cached 0MB) "
    "CPU [12%@1728,8%@1728,5%@1728,4%@1728,6%@1728,3%@1728] EMC_FREQ 0% GR3D_FREQ 45% "
    "cpu@52.5C soc0@50.5C soc1@49.5C tj@52.5C "
    "VDD_IN 5123mW/4980mW VDD_CPU_GPU_CV 1890mW/1750mW VDD_SOC 1234mW/1200mW";

TEST(BenchPower, ParsesInstantaneousRailPower) {
  auto in = bench::parse_rail_mw(kTegraLine, "VDD_IN");
  ASSERT_TRUE(in.has_value());
  EXPECT_DOUBLE_EQ(*in, 5123.0) << "must take the instantaneous value, not the trailing average";

  auto gpu = bench::parse_rail_mw(kTegraLine, "VDD_CPU_GPU_CV");
  ASSERT_TRUE(gpu.has_value());
  EXPECT_DOUBLE_EQ(*gpu, 1890.0);
}

TEST(BenchPower, ReturnsNothingForAbsentOrMalformedRail) {
  EXPECT_FALSE(bench::parse_rail_mw(kTegraLine, "VDD_NOT_A_RAIL").has_value());
  EXPECT_FALSE(bench::parse_rail_mw("RAM 100/200MB", "VDD_IN").has_value());
  // A number that is not a milliwatt field must not be mistaken for one.
  EXPECT_FALSE(bench::parse_rail_mw("VDD_IN 45%", "VDD_IN").has_value());
}

TEST(BenchPower, AveragesAcrossTheCapturedWindow) {
  std::vector<std::string> lines = {
      "x VDD_IN 1000mW/900mW y",
      "x VDD_IN 3000mW/900mW y",
      "x RAM 1/2MB y",  // no rail: skipped, not counted as zero
      "x VDD_IN 2000mW/900mW y",
  };
  auto s = bench::summarize_log(lines, "VDD_IN");
  EXPECT_TRUE(s.available);
  EXPECT_EQ(s.samples, 3u);
  EXPECT_DOUBLE_EQ(s.mean_mw, 2000.0);
  EXPECT_DOUBLE_EQ(s.max_mw, 3000.0);
  EXPECT_DOUBLE_EQ(s.mean_w(), 2.0);
}

// Off a Jetson there is nothing to sample; the summary must say so rather
// than reporting a confident zero.
TEST(BenchPower, UnavailableWhenNoRailsPresent) {
  auto s = bench::summarize_log({"RAM 1/2MB", "CPU [1%@100]"}, "VDD_IN");
  EXPECT_FALSE(s.available);
  EXPECT_EQ(s.samples, 0u);
}
