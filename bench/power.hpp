#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace bench {

// Board power over the measure window, from tegrastats.
//
// This is the column that makes the Orin numbers worth publishing: a
// latency table alone invites "so what, a desktop GPU is faster", whereas
// Hz/W is the comparison an embedded controller is actually judged on.
//
// Off a Jetson there is no tegrastats and no sensible substitute, so the
// sampler reports unavailable and the CSV carries an empty watts field
// rather than a fabricated one.

struct PowerSummary {
  bool available = false;
  double mean_mw = 0.0;
  double max_mw = 0.0;
  std::size_t samples = 0;

  double mean_w() const { return mean_mw / 1000.0; }
};

// Pulls one rail's instantaneous milliwatts out of a tegrastats line.
//
// The rails appear as `NAME <inst>mW/<avg>mW`; we want the instantaneous
// figure because the trailing average tegrastats reports is over its own
// window, not ours. Rail naming differs across JetPack releases and carrier
// boards, hence the caller-supplied name and the empty return when it is
// absent.
inline std::optional<double> parse_rail_mw(const std::string& line, const std::string& rail) {
  std::size_t pos = line.find(rail);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  pos += rail.size();
  while (pos < line.size() && line[pos] == ' ') {
    ++pos;
  }
  std::size_t start = pos;
  while (pos < line.size() && (std::isdigit(static_cast<unsigned char>(line[pos])) != 0)) {
    ++pos;
  }
  if (pos == start) {
    return std::nullopt;
  }
  // Must be the milliwatt field, not some other number that happens to
  // follow the rail name.
  if (line.compare(pos, 2, "mW") != 0) {
    return std::nullopt;
  }
  return std::stod(line.substr(start, pos - start));
}

// Averages a rail across a captured tegrastats log.
inline PowerSummary summarize_log(const std::vector<std::string>& lines, const std::string& rail) {
  PowerSummary s;
  double sum = 0.0;
  for (const auto& line : lines) {
    auto mw = parse_rail_mw(line, rail);
    if (!mw.has_value()) {
      continue;
    }
    sum += *mw;
    s.max_mw = std::max(s.max_mw, *mw);
    ++s.samples;
  }
  if (s.samples > 0) {
    s.available = true;
    s.mean_mw = sum / static_cast<double>(s.samples);
  }
  return s;
}

// Runs tegrastats for the duration of the measure window. tegrastats manages
// its own lifecycle through `--stop`, so there is no PID to track.
class PowerSampler {
 public:
  explicit PowerSampler(std::string rail = "VDD_IN", std::string log_path = "/tmp/mppi_tegra.log")
      : rail_(std::move(rail)), log_path_(std::move(log_path)) {}

  static bool tegrastats_available() {
    return std::system("command -v tegrastats > /dev/null 2>&1") == 0;
  }

  // Returns false when there is nothing to sample; callers should carry on
  // and leave the power column empty.
  bool start(int interval_ms = 100) {
    if (!tegrastats_available()) {
      return false;
    }
    std::system("tegrastats --stop > /dev/null 2>&1");
    std::remove(log_path_.c_str());
    std::string cmd = "tegrastats --interval " + std::to_string(interval_ms) + " --logfile " +
                      log_path_ + " > /dev/null 2>&1 &";
    running_ = (std::system(cmd.c_str()) == 0);
    return running_;
  }

  PowerSummary stop() {
    if (!running_) {
      return {};
    }
    std::system("tegrastats --stop > /dev/null 2>&1");
    running_ = false;

    std::vector<std::string> lines;
    std::ifstream in(log_path_);
    std::string line;
    while (std::getline(in, line)) {
      lines.push_back(line);
    }
    return summarize_log(lines, rail_);
  }

 private:
  std::string rail_;
  std::string log_path_;
  bool running_ = false;
};

}  // namespace bench
