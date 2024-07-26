cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(kernel_comp)

enable_language(CUDA)

add_executable(kernel_benchmark "main.cu")

set(CUDA_ARCH "native")
set_target_properties(kernel_benchmark PROPERTIES 
		CUDA_ARCHITECTURES ${CUDA_ARCH} 
		CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(kernel_benchmark PRIVATE CUDA::toolkit)
