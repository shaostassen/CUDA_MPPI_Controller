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

CLANG="${CLANGXX:-clang++}"

# Fail loudly if there is no compiler to run. Without this the loop below would
# report every file "ok" and exit 0, because a missing clang++ produces no
# diagnostics to find — a gate that silently passes is worse than no gate, and
# this script runs in CI.
if ! command -v "$CLANG" >/dev/null 2>&1; then
  echo "ERROR: '$CLANG' not found on PATH; cannot syntax-check CUDA sources." >&2
  echo "Install clang (>= 14 parses CUDA) or set CLANGXX to its path." >&2
  exit 2
fi

# googletest headers, if a CPU build tree exists (needed by the .cu tests).
# Configuring is enough to populate them: FetchContent downloads at configure
# time, so CI can run `cmake --preset cpu` without building.
GTEST_INC="$ROOT/build/cpu/_deps/googletest-src/googletest/include"
GTEST_FLAG=""
[ -d "$GTEST_INC" ] && GTEST_FLAG="-I$GTEST_INC"

FILES=$(find "$ROOT/src" "$ROOT/tests" "$ROOT/examples" -name '*.cu' 2>/dev/null | sort)
if [ -z "$FILES" ]; then
  echo "no .cu sources found"
  exit 0
fi

status=0
for f in $FILES; do
  rel="${f#"$ROOT"/}"
  # Keep clang's exit status as the verdict rather than grepping its output for
  # the word "error": the status is set by the compiler itself, so it cannot be
  # faked by a diagnostic that happens to be worded differently, nor missed when
  # clang fails to run at all. The greps below only tidy up what gets printed.
  out=$("$CLANG" -x cuda --cuda-device-only -nocudainc -nocudalib \
    --cuda-gpu-arch="$ARCH" -std=c++20 -fsyntax-only \
    -include "$STUB/cuda_runtime.h" \
    -I"$STUB" -I"$ROOT/include" -I"$ROOT/examples" $GTEST_FLAG \
    "$f" 2>&1)
  rc=$?
  out=$(printf '%s\n' "$out" | grep -v "gtest-printers.h" | grep -vE "^\s*(524|\s*\|)" \
    | grep -vE "In file included from|1 warning generated")
  if [ "$rc" -ne 0 ]; then
    echo "FAIL  $rel"
    echo "$out" | head -20
    status=1
  else
    echo "ok    $rel"
  fi
done

exit $status
