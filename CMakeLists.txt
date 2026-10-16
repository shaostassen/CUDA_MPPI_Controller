cmake_minimum_required(VERSION 3.28)
project(cuda_mppi_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Build CUDA kernels (requires nvcc; targets Jetson Orin sm_87)" OFF)

if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "ENABLE_CUDA=ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "87-real;87-virtual")
  # cuBLAS for the S2.3 weighted-noise Sgemv; CUB is header-only and ships
  # with the toolkit.
  find_package(CUDAToolkit REQUIRED)
endif()

# Optional: enables the S1.6 OpenMP execution policy, tests, and bench.
# On macOS: brew install libomp, configure with -DOpenMP_ROOT=$(brew --prefix libomp).
find_package(OpenMP COMPONENTS CXX)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# JSON config files for the S3.1 benchmark harness. Release archive rather
# than the git repo: the repo clone is large and none of it is needed.
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

enable_testing()

add_library(mppi_core INTERFACE)
target_include_directories(mppi_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mppi_core INTERFACE cxx_std_20)

if(ENABLE_CUDA)
  add_subdirectory(src/cuda)
endif()

add_subdirectory(bench)
add_subdirectory(examples)
add_subdirectory(tests)
