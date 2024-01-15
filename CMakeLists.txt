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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

set(CMAKE_CUDA_ARCHITECTURES 70)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


macro (add_cuda_exec exec_name exec_file)
    # GPU代码
    add_executable(${exec_name} ${exec_file})
    set_target_properties(${exec_name} PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${exec_name}
        PUBLIC cudart /home/huanghy/progs/nvidia_hpc_sdk/Linux_x86_64/22.11/math_libs/lib64/libcusparse.so
    )
endmacro (add_cuda_exec)

add_cuda_exec(spmv main.cu)
add_cuda_exec(tmp tmp.cu)
add_cuda_exec(genl genl.cu)