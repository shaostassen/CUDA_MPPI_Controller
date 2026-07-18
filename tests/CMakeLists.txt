include(GoogleTest)

add_executable(mppi_tests toolchain_smoke_test.cpp)
target_link_libraries(mppi_tests PRIVATE mppi_core GTest::gtest_main)

if(ENABLE_CUDA)
  target_sources(mppi_tests PRIVATE cuda_smoke_test.cu)
  target_link_libraries(mppi_tests PRIVATE mppi_cuda)
endif()

gtest_discover_tests(mppi_tests)
