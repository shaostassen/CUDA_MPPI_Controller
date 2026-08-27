// Stand-in for CUB, for syntax checking only. Only the DeviceReduce entry
// points this project uses are declared. See ../cuda_runtime.h for caveats.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cub {

struct DeviceReduce {
  template <typename InputIteratorT, typename OutputIteratorT>
  static cudaError_t Min(void* d_temp_storage, std::size_t& temp_storage_bytes,
                         InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                         cudaStream_t stream = 0) {
    (void)d_temp_storage;
    (void)d_in;
    (void)d_out;
    (void)num_items;
    (void)stream;
    temp_storage_bytes = 1024;
    return cudaSuccess;
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  static cudaError_t Sum(void* d_temp_storage, std::size_t& temp_storage_bytes,
                         InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                         cudaStream_t stream = 0) {
    (void)d_temp_storage;
    (void)d_in;
    (void)d_out;
    (void)num_items;
    (void)stream;
    temp_storage_bytes = 1024;
    return cudaSuccess;
  }
};

}  // namespace cub
