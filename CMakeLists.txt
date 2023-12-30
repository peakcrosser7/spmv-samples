cmake_minimum_required(VERSION 3.20)

project(spmv_samples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    # set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(src)

# add_executable(ctmp ctmp.cpp)