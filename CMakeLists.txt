cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)

project(kernel_comp)

OPTION(BUILD_CUDA "Build cuda version" ON)
OPTION(BUILD_HIP "Build hip version" OFF)
OPTION(CUDA_ARCH "Set GPU arch for CUDA manually" "native")
OPTION(HIP_ARCH "Set GPU arch for HIP manually" "")

if (BUILD_CUDA)
	enable_language(CUDA)

	add_executable(cuda_benchmark "main.cu")

	set(CUDA_ARCH "native")
	set_target_properties(cuda_benchmark PROPERTIES 
			CUDA_ARCHITECTURES ${CUDA_ARCH} 
			CUDA_RESOLVE_DEVICE_SYMBOLS ON)

	find_package(CUDAToolkit REQUIRED)
	target_link_libraries(cuda_benchmark PRIVATE CUDA::toolkit)
endif()

if (BUILD_HIP)
	enable_language(HIP)

	add_executable(hip_benchmark "main.hip")

	find_package(hip REQUIRED)
	target_link_libraries(hip_benchmark PRIVATE hip::host)
	if (not "${HIP_ARCH}" STREQUAL "")
		set_property(TARGET hip_benchmark PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
	endif()
endif()