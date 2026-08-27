add_executable(lqr_cross_check lqr_cross_check.cpp)
target_link_libraries(lqr_cross_check PRIVATE mppi_core)
target_include_directories(lqr_cross_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cartpole_replay cartpole_replay.cpp)
target_link_libraries(cartpole_replay PRIVATE mppi_core)
target_include_directories(cartpole_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
  add_executable(omp_bench omp_bench.cpp)
  target_link_libraries(omp_bench PRIVATE mppi_core OpenMP::OpenMP_CXX)
  target_include_directories(omp_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
