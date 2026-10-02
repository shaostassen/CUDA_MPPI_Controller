#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

// One row of Table 1: a (K, T) point plus which backends to time at it.
// Defaults match the §4 protocol so a config only has to state what differs.
struct BenchConfig {
  std::string name;
  int samples = 16384;  // K
  int horizon = 50;     // T
  double dt = 0.02;
  double lambda = 2.0;
  double sigma = 3.0;
  double u_limit = 15.0;
  int warmup = 50;
  int measure = 500;
  unsigned seed = 0;
  std::string power_rail = "VDD_IN";
  std::vector<std::string> backends{"cpu_1t", "cpu_omp", "cuda_v1"};
};

// Separated from file loading so the schema can be tested on strings.
inline BenchConfig parse_config(const std::string& text) {
  auto j = nlohmann::json::parse(text);
  BenchConfig c;
  c.name = j.value("name", std::string("unnamed"));
  c.samples = j.value("samples", c.samples);
  c.horizon = j.value("horizon", c.horizon);
  c.dt = j.value("dt", c.dt);
  c.lambda = j.value("lambda", c.lambda);
  c.sigma = j.value("sigma", c.sigma);
  c.u_limit = j.value("u_limit", c.u_limit);
  c.warmup = j.value("warmup", c.warmup);
  c.measure = j.value("measure", c.measure);
  c.seed = j.value("seed", c.seed);
  c.power_rail = j.value("power_rail", c.power_rail);
  if (j.contains("backends")) {
    c.backends = j.at("backends").get<std::vector<std::string>>();
  }

  if (c.samples <= 0 || c.horizon <= 0) {
    throw std::runtime_error("config '" + c.name + "': samples and horizon must be positive");
  }
  if (c.measure <= 0) {
    throw std::runtime_error("config '" + c.name + "': measure must be positive");
  }
  if (c.sigma <= 0.0) {
    throw std::runtime_error("config '" + c.name + "': sigma must be positive");
  }
  return c;
}

inline BenchConfig load_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open config: " + path);
  }
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto c = parse_config(text);
  if (c.name == "unnamed") {
    // Fall back to the filename stem so CSV rows stay identifiable.
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    c.name = path.substr(slash == std::string::npos ? 0 : slash + 1,
                         dot == std::string::npos ? std::string::npos : dot - slash - 1);
  }
  return c;
}

}  // namespace bench
