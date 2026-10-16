# Environment (Jetson AGX Orin 64 GB)

Filled in during S0.1 (toolchain verification). Every number below should
come from a command actually run on the Orin — no placeholders left in once
that story is closed.

| Item | Value | Command |
|---|---|---|
| JetPack version | | `apt show nvidia-jetpack` |
| L4T version | | `head -1 /etc/nv_tegra_release` |
| CUDA toolkit | | `nvcc --version` |
| Driver | | `nvidia-smi` (or `cat /proc/driver/nvidia/version`) |
| Compute capability | | `deviceQuery` → should report sm_87 |
| Power mode | | `nvpmodel -q` |
| Fan profile | | |
| cmake version | | `cmake --version` (≥ 3.28 required; `pip install cmake` if apt's is older) |
| ncu | works under sudo? | `sudo ncu --version` |
| nsys | | `nsys --version` |

## Mac → Orin dev flow

- [ ] SSH key added to the Orin's `~/.ssh/authorized_keys`
- [ ] VS Code Remote-SSH (or rsync-on-save) confirmed round-tripping edits
- [ ] `scripts/gpu-test.sh` runs clean end to end
