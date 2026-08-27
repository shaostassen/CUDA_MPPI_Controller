#!/usr/bin/env bash
# Syntax- and semantic-check the .cu/.cuh sources WITHOUT a CUDA toolkit.
#
# Why this exists: the only CUDA hardware for this project is a remote Jetson,
# so the edit->compile loop for kernels would otherwise be an SSH round trip.
# clang can parse CUDA without nvcc, and with the minimal API stub in
# scripts/cudastub/ that is enough to catch the ordinary authoring mistakes —
# template instantiation failures, wrong member names, host/device attribute
# errors, bad launch syntax.
#
# What it does NOT do:
#   * prove nvcc accepts the code — the stub mirrors the real cuda_runtime.h
#     signatures but is not it, and nvcc has its own extensions and quirks
#   * generate any device code (this machine has no NVPTX backend)
#   * say anything about runtime behavior, numerics, or performance
#
# A clean run means "worth pushing to the Orin", not "verified".
#
# Usage: scripts/cuda-syntax-check.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUB="$ROOT/scripts/cudastub"
ARCH="${CUDA_ARCH:-sm_87}"

# googletest headers, if a CPU build tree exists (needed by the .cu tests).
GTEST_INC="$ROOT/build/cpu/_deps/googletest-src/googletest/include"
GTEST_FLAG=""
[ -d "$GTEST_INC" ] && GTEST_FLAG="-I$GTEST_INC"

FILES=$(find "$ROOT/src" "$ROOT/tests" -name '*.cu' 2>/dev/null | sort)
if [ -z "$FILES" ]; then
  echo "no .cu sources found"
  exit 0
fi

status=0
for f in $FILES; do
  rel="${f#"$ROOT"/}"
  out=$(clang++ -x cuda --cuda-device-only -nocudainc -nocudalib \
    --cuda-gpu-arch="$ARCH" -std=c++20 -fsyntax-only \
    -include "$STUB/cuda_runtime.h" \
    -I"$STUB" -I"$ROOT/include" -I"$ROOT/examples" $GTEST_FLAG \
    "$f" 2>&1 | grep -v "gtest-printers.h" | grep -vE "^\s*(524|\s*\|)" \
    | grep -vE "In file included from|1 warning generated")
  if echo "$out" | grep -q "error"; then
    echo "FAIL  $rel"
    echo "$out" | head -20
    status=1
  else
    echo "ok    $rel"
  fi
done

exit $status
