#!/usr/bin/env bash
# S0.1: verify the Jetson toolchain and write docs/env.md from what is
# actually installed. Run ON the Orin, from the repo root:
#
#   scripts/jetson-setup-check.sh
#
# Records only what it observes — anything it cannot determine is written as
# "not detected" rather than guessed, because env.md exists to make the
# benchmark numbers reproducible and a plausible-looking wrong version
# defeats the point.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/env.md"
FAILED=0

have() { command -v "$1" > /dev/null 2>&1; }
note_fail() { echo "  MISSING: $1"; FAILED=1; }

echo "=== S0.1 toolchain verification ==="

# --- identity -------------------------------------------------------------
MODEL="not detected"
[ -f /proc/device-tree/model ] && MODEL="$(tr -d '\0' < /proc/device-tree/model)"
L4T="not detected"
[ -f /etc/nv_tegra_release ] && L4T="$(head -1 /etc/nv_tegra_release)"
# Installed state, not `apt show`: that reports the repo's candidate version
# whether or not the package is installed, which would record a JetPack that
# is not on the machine.
JETPACK="not detected"
if have dpkg-query; then
  JETPACK="nvidia-jetpack meta-package not installed (components may be installed individually)"
  JP_STATE="$(dpkg-query -W -f='${db:Status-Status} ${Version}' nvidia-jetpack 2>/dev/null)"
  case "$JP_STATE" in
    "installed "*) JETPACK="${JP_STATE#installed }" ;;
  esac
fi
OSREL="$( (grep PRETTY_NAME /etc/os-release 2>/dev/null || echo '') | cut -d'"' -f2)"
[ -z "$OSREL" ] && OSREL="$(uname -sr)"

echo "model:   $MODEL"
echo "l4t:     $L4T"
echo "jetpack: $JETPACK"

if [ "$MODEL" = "not detected" ]; then
  echo "  WARNING: no /proc/device-tree/model — this does not look like a Jetson."
fi

# --- toolchain ------------------------------------------------------------
NVCC="not detected"
if have nvcc; then
  NVCC="$(nvcc --version | awk '/release/{print $0}' | sed 's/^ *//')"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  # JetPack installs it here even when PATH was never set up.
  echo "  note: nvcc not on PATH but /usr/local/cuda/bin/nvcc exists"
  echo "        add: export PATH=/usr/local/cuda/bin:\$PATH"
  NVCC="$(/usr/local/cuda/bin/nvcc --version | awk '/release/{print $0}' | sed 's/^ *//')"
else
  note_fail "nvcc (install cuda-toolkit, or add /usr/local/cuda/bin to PATH)"
fi

CMAKE="not detected"
if have cmake; then
  CMAKE="$(cmake --version | head -1 | awk '{print $3}')"
  # CMakePresets v6 and the preset flow need >= 3.28.
  if [ "$(printf '%s\n3.28.0\n' "$CMAKE" | sort -V | head -1)" != "3.28.0" ]; then
    echo "  WARNING: cmake $CMAKE < 3.28 — install a newer one (pipx install cmake)"
    FAILED=1
  fi
else
  note_fail "cmake"
fi

GXX="$(g++ --version 2>/dev/null | head -1)"
[ -z "$GXX" ] && GXX="not detected"

# --- profilers ------------------------------------------------------------
NSYS="not detected"
have nsys && NSYS="$(nsys --version 2>/dev/null | head -1)"
[ "$NSYS" = "not detected" ] && echo "  note: nsys absent — S3.4's system-level profiling needs it"

NCU="not detected"
NCU_COUNTERS="not tested"
if have ncu; then
  NCU="$(ncu --version 2>/dev/null | grep -m1 -i version || echo present)"
  # Counter collection on Tegra normally requires root; find out now rather
  # than when S3.4 is blocked on it.
  if sudo -n true 2>/dev/null; then
    if sudo ncu --version > /dev/null 2>&1; then
      NCU_COUNTERS="sudo works (profile under sudo)"
    else
      NCU_COUNTERS="sudo ncu failed"
    fi
  else
    NCU_COUNTERS="needs password-prompted sudo; run 'sudo ncu --version' to confirm"
  fi
else
  echo "  note: ncu absent — S3.4's kernel profiling needs it"
fi

# --- power / clocks -------------------------------------------------------
NVPMODEL="not detected"
have nvpmodel && NVPMODEL="$(sudo -n nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo 'present (needs sudo to query)')"
CLOCKS="not detected"
have jetson_clocks && CLOCKS="present"
TEGRASTATS="not detected"
have tegrastats && TEGRASTATS="present"
[ "$TEGRASTATS" = "not detected" ] && echo "  note: tegrastats absent — the benchmark power column will stay empty"

# --- device properties ----------------------------------------------------
DEVQ="not run (device_query not built)"
DEVQ_BIN=""
for c in "$ROOT/build/release/examples/device_query" "$ROOT/build/cpu/examples/device_query"; do
  [ -x "$c" ] && DEVQ_BIN="$c" && break
done
if [ -n "$DEVQ_BIN" ]; then
  DEVQ="$("$DEVQ_BIN" 2>&1)"
else
  echo "  note: device_query not built yet — run:"
  echo "        cmake --preset release && cmake --build --preset release"
  echo "        then re-run this script to capture device properties"
fi

CAP="$(printf '%s\n' "$DEVQ" | awk -F': ' '/device_0_compute_capability/{print $2}')"
if [ -n "$CAP" ]; then
  echo "compute capability: $CAP"
  if [ "$CAP" != "8.7" ]; then
    echo "  WARNING: expected 8.7 (sm_87) for Orin — CMAKE_CUDA_ARCHITECTURES may need changing"
  fi
fi

# --- write env.md ---------------------------------------------------------
mkdir -p "$(dirname "$OUT")"
{
  echo "# Environment (Jetson Orin)"
  echo
  echo "Generated by \`scripts/jetson-setup-check.sh\` on $(date -Iseconds)."
  echo "Everything below was read off the machine; fields it could not"
  echo "determine say \"not detected\" rather than carrying a guess."
  echo
  echo "| Item | Value |"
  echo "|---|---|"
  echo "| Model | $MODEL |"
  echo "| OS | $OSREL |"
  echo "| L4T | $L4T |"
  echo "| JetPack | $JETPACK |"
  echo "| nvcc | $NVCC |"
  echo "| g++ | $GXX |"
  echo "| cmake | $CMAKE |"
  echo "| nsys | $NSYS |"
  echo "| ncu | $NCU |"
  echo "| ncu counters | $NCU_COUNTERS |"
  echo "| nvpmodel | $NVPMODEL |"
  echo "| jetson_clocks | $CLOCKS |"
  echo "| tegrastats | $TEGRASTATS |"
  echo
  echo '## Device properties'
  echo
  echo '```'
  printf '%s\n' "$DEVQ"
  echo '```'
  echo
  echo '## Benchmark protocol on this machine'
  echo
  echo 'Lock clocks and fix the power mode before recording anything'
  echo '(PLAN.md §4). `scripts/run_table1.sh` applies jetson_clocks and'
  echo 'records what it did alongside the CSV.'
  echo
  echo '```'
  echo 'sudo nvpmodel -m 0        # MAXN'
  echo 'sudo jetson_clocks'
  echo 'scripts/run_table1.sh'
  echo '```'
} > "$OUT"

echo
echo "wrote $OUT"
if [ "$FAILED" -ne 0 ]; then
  echo "S0.1: INCOMPLETE — see MISSING/WARNING lines above"
  exit 1
fi
echo "S0.1: toolchain checks passed"
