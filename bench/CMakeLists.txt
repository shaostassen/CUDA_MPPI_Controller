# The harness compiles as CUDA when the toolchain is present, so the GPU
# backends can be timed from the same binary as the CPU ones. Without CUDA it
# is an ordinary C++ target and the cuda_* backends report as unavailable
# rather than silently vanishing from the table.
if(ENABLE_CUDA)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(mppi_bench main.cpp)
target_link_libraries(mppi_bench PRIVATE mppi_core nlohmann_json::nlohmann_json)
target_include_directories(mppi_bench PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/examples
)

if(ENABLE_CUDA)
  target_link_libraries(mppi_bench PRIVATE mppi_cuda)
  target_compile_definitions(mppi_bench PRIVATE MPPI_BENCH_WITH_CUDA)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(mppi_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
