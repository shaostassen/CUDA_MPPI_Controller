#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bench {

// Latency summary for one benchmark point. p99 matters as much as the mean
// here: a controller that averages 5 ms but spikes to 40 ms misses its
// deadline, and Table 2's sustainable-rate column is decided by the tail.
struct Stats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  std::size_t count = 0;
};

// Nearest-rank percentile: the smallest sample at or above the requested
// fraction of the ordered set. Chosen over interpolation because it always
// reports a latency that was actually observed.
inline double percentile_sorted(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) {
    return 0.0;
  }
  double rank = std::ceil(p * static_cast<double>(sorted.size()));
  std::size_t idx = static_cast<std::size_t>(rank) - 1;
  if (rank < 1.0) {
    idx = 0;
  }
  idx = std::min(idx, sorted.size() - 1);
  return sorted[idx];
}

inline Stats summarize(std::vector<double> samples) {
  Stats s;
  if (samples.empty()) {
    return s;
  }
  std::sort(samples.begin(), samples.end());
  double sum = 0.0;
  for (double v : samples) {
    sum += v;
  }
  s.count = samples.size();
  s.mean_ms = sum / static_cast<double>(samples.size());
  s.p50_ms = percentile_sorted(samples, 0.50);
  s.p99_ms = percentile_sorted(samples, 0.99);
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  return s;
}

}  // namespace bench
