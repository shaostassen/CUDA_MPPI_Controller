#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "mppi/cuda/check.cuh"

namespace mppi::cuda {

// Move-only RAII device allocation. Deliberately minimal: the point of S2.1
// is that the steady-state loop performs no allocation and no per-iteration
// transfers, so this type exists to make ownership obvious at setup time,
// not to be clever.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void reset() {
    if (ptr_ != nullptr) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  void upload(const T* host, std::size_t count) {
    CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void download(T* host, std::size_t count) const {
    CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// Page-locked host allocation, mapped into the device address space.
//
// On the Orin this matters more than on a discrete GPU: CPU and GPU share
// one physical DRAM, so a mapped pinned buffer lets the closed loop hand
// over x0 (and read back u0) with no copy at all, which is the difference
// between "no memcpy in the steady-state loop" being true and being
// approximately true. `device_ptr()` is what kernels should take.
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;

  explicit PinnedBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaHostAlloc(&host_, count_ * sizeof(T), cudaHostAllocMapped));
      CUDA_CHECK(cudaHostGetDevicePointer(&device_, host_, 0));
    }
  }

  ~PinnedBuffer() { reset(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept
      : host_(std::exchange(other.host_, nullptr)),
        device_(std::exchange(other.device_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      host_ = std::exchange(other.host_, nullptr);
      device_ = std::exchange(other.device_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void reset() {
    if (host_ != nullptr) {
      CUDA_CHECK(cudaFreeHost(host_));
      host_ = nullptr;
      device_ = nullptr;
    }
    count_ = 0;
  }

  T* host_ptr() { return host_; }
  const T* host_ptr() const { return host_; }
  T* device_ptr() { return device_; }
  std::size_t size() const { return count_; }

 private:
  T* host_ = nullptr;
  T* device_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace mppi::cuda
