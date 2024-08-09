cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX CUDA)

# C++ and CUDA Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# Setup pybind11
set(PYBIND11_NEWPYTHON ON)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_LIST_DIR}/csrc)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add selective_scan_jax_cuda module
pybind11_add_module(
  selective_scan_jax_cuda
  ${CMAKE_CURRENT_LIST_DIR}/csrc/kernels.cc.cu
  ${CMAKE_CURRENT_LIST_DIR}/csrc/selective_scan_jax_cuda.cc
)

# Install the selective_scan_jax_cuda module
install(TARGETS selective_scan_jax_cuda LIBRARY DESTINATION .)
target_link_libraries(selective_scan_jax_cuda PUBLIC cublas)