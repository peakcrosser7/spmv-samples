
macro (add_cuda_exec exec_name exec_file)
    # GPU代码
    add_executable(${exec_name} ${exec_file})
    # target_link_libraries(${exec_name} PRIVATE graph_genlx)
    set_target_properties(${exec_name} PROPERTIES 
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(spmv
        PUBLIC cudart /home/huanghy/progs/nvidia_hpc_sdk/Linux_x86_64/22.11/math_libs/lib64/libcusparse.so
    )
endmacro (add_cuda_exec)

add_cuda_exec(spmv main.cu)