include(GoogleTest)

add_executable(mppi_tests toolchain_smoke_test.cpp dynamics_test.cpp philox_test.cpp
               costs_test.cpp mppi_cartpole_test.cpp lqr_test.cpp replay_test.cpp
               layout_test.cpp)
target_link_libraries(mppi_tests PRIVATE mppi_core GTest::gtest_main)
target_include_directories(mppi_tests PRIVATE ${CMAKE_SOURCE_DIR}/examples)

if(OpenMP_CXX_FOUND)
  target_sources(mppi_tests PRIVATE omp_test.cpp)
  target_link_libraries(mppi_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
  target_sources(mppi_tests PRIVATE cuda_smoke_test.cu philox_parity_test.cu cuda_rollout_test.cu)
  target_link_libraries(mppi_tests PRIVATE mppi_cuda)
endif()

gtest_discover_tests(mppi_tests)
