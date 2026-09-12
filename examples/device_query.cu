// S0.1: reports the device facts docs/env.md has to record.
//
// JetPack 6 no longer ships the cuda-samples deviceQuery binary, and the
// numbers here are ones the project actually depends on — compute
// capability must be 8.7 or the fatbin targets the wrong arch, and the
// integrated/unified-memory flags are what justify the zero-copy handoff in
// device_buffers.cuh.
#include <cuda_runtime.h>

#include <cstdio>

int main() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  if (count == 0) {
    std::fprintf(stderr, "no CUDA devices found\n");
    return 1;
  }

  int runtime_version = 0, driver_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDriverGetVersion(&driver_version);
  std::printf("cuda_runtime: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  std::printf("cuda_driver: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
  std::printf("device_count: %d\n", count);

  for (int i = 0; i < count; ++i) {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, i);
    std::printf("device_%d_name: %s\n", i, p.name);
    std::printf("device_%d_compute_capability: %d.%d\n", i, p.major, p.minor);
    std::printf("device_%d_sm_arch: sm_%d%d\n", i, p.major, p.minor);
    std::printf("device_%d_multiprocessors: %d\n", i, p.multiProcessorCount);
    std::printf("device_%d_clock_khz: %d\n", i, p.clockRate);
    std::printf("device_%d_global_mem_mb: %.0f\n", i,
                static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0));
    std::printf("device_%d_l2_cache_kb: %d\n", i, p.l2CacheSize / 1024);
    std::printf("device_%d_shared_mem_per_block_kb: %d\n", i,
                static_cast<int>(p.sharedMemPerBlock / 1024));
    std::printf("device_%d_regs_per_block: %d\n", i, p.regsPerBlock);
    std::printf("device_%d_warp_size: %d\n", i, p.warpSize);
    std::printf("device_%d_max_threads_per_block: %d\n", i, p.maxThreadsPerBlock);
    std::printf("device_%d_max_threads_per_sm: %d\n", i, p.maxThreadsPerMultiProcessor);
    std::printf("device_%d_memory_bus_width_bits: %d\n", i, p.memoryBusWidth);
    std::printf("device_%d_memory_clock_khz: %d\n", i, p.memoryClockRate);
    // Peak theoretical bandwidth: DDR, so two transfers per clock.
    double bw = 2.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8.0) / 1.0e9;
    std::printf("device_%d_peak_bandwidth_gbps: %.1f\n", i, bw);
    // Integrated == shares DRAM with the CPU, which is the whole reason the
    // closed loop can hand over x0 without a copy.
    std::printf("device_%d_integrated: %d\n", i, p.integrated);
    std::printf("device_%d_can_map_host_memory: %d\n", i, p.canMapHostMemory);
    std::printf("device_%d_unified_addressing: %d\n", i, p.unifiedAddressing);
    std::printf("device_%d_concurrent_kernels: %d\n", i, p.concurrentKernels);
    std::printf("device_%d_cooperative_launch: %d\n", i, p.cooperativeLaunch);
  }
  return 0;
}
