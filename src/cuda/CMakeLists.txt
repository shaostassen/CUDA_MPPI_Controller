add_library(mppi_cuda STATIC
  smoke.cu
)
target_link_libraries(mppi_cuda PUBLIC mppi_core)
set_target_properties(mppi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
