#!/usr/bin/env bash
# Run on the Jetson Orin — builds and tests the CUDA path.
# CI only covers the CPU-only preset; this is the local GPU counterpart.
set -euo pipefail

cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
