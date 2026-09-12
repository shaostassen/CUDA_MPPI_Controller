add_executable(lqr_cross_check lqr_cross_check.cpp)
target_link_libraries(lqr_cross_check PRIVATE mppi_core)
target_include_directories(lqr_cross_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cartpole_replay cartpole_replay.cpp)
target_link_libraries(cartpole_replay PRIVATE mppi_core)
target_include_directories(cartpole_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_CUDA)
  add_executable(device_query device_query.cu)
  target_link_libraries(device_query PRIVATE mppi_core)

  add_executable(cuda_reduction_bench cuda_reduction_bench.cu)
  target_link_libraries(cuda_reduction_bench PRIVATE mppi_core mppi_cuda)
  target_include_directories(cuda_reduction_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(OpenMP_CXX_FOUND)
  add_executable(omp_bench omp_bench.cpp)
  target_link_libraries(omp_bench PRIVATE mppi_core OpenMP::OpenMP_CXX)
  target_include_directories(omp_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
