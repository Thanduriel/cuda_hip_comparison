#include <vector>
#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

/// 32 bit should be faster on GPU
using IndexType = uint32_t;
// julia is column major
constexpr bool COLUMN_MAJOR = true;
// prevent aliasing
#define WITH_RESTRICT
#ifdef WITH_RESTRICT
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


__device__ __host__ IndexType flatIdx(IndexType x, IndexType y, IndexType sizeX, IndexType sizeY) {
	if constexpr (COLUMN_MAJOR){
		return x + y * sizeX;
	} else {
		return x * sizeY + y;
	} 
}

using T = float;
//template<typename T> 
__global__ void myKernel(RESTRICT T* out, RESTRICT const T* arr, IndexType sizeX, IndexType sizeY){
	const IndexType i = (blockIdx.x * blockDim.x) + threadIdx.x;
	const IndexType j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i > sizeX || j > sizeY){
		return;
	}

	for (IndexType k = 0; k < sizeX; ++k){
	//	out[flatIdx(k, i, sizeX, sizeY)] += arr[flatIdx(i, j, sizeX, sizeY)];
		atomicAdd(&out[flatIdx(k, i, sizeX, sizeY)], arr[flatIdx(i, j, sizeX, sizeY)]);
	}
}

//template<typename T> 
void run_benchmark(IndexType sizeX, IndexType sizeY) {
	const IndexType numEl = sizeX * sizeY;
	// prepare input data
	std::vector<T> hostBuf(numEl, 0.0);
	auto setValue = [&](IndexType beginX, IndexType endX, IndexType beginY, IndexType endY, T value){
		for(IndexType iy = beginY; iy < endY; ++iy){
			for(IndexType ix = beginX; ix < endX; ++ix){
				hostBuf[flatIdx(ix, iy, sizeX, sizeY)] = value;
			}
		}
	};
	setValue(100, 200, 100, 200, 1.0);
	setValue(350, 450, 350, 450, 1.0);

	std::vector<T> resultCPU(numEl, 0.0);
	for(IndexType iy = 0; iy < sizeY; ++iy){
		for(IndexType ix = 0; ix < sizeX; ++ix){
			for(IndexType ik = 0; ik < sizeX; ++ik){
				resultCPU[flatIdx(ik, ix, sizeX, sizeY)] += hostBuf[flatIdx(ix, iy, sizeX, sizeY)];
			}
		}
	}

	// initialize gpu buffers
	const size_t bytes = hostBuf.size() * sizeof(T);
	T* arr = nullptr;
	T* out = nullptr;
	cudaMalloc(&arr, bytes);
	cudaMalloc(&out, bytes);

	cudaMemcpy(arr, hostBuf.data(), bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemset(out, 0, bytes);

	// determine block size
	int maxBlockSize = 0;
    int minGridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, myKernel, 0, 0);
	const unsigned blockDim = std::sqrt(maxBlockSize);
	const dim3 threadsPerBlock(blockDim, blockDim);
	const dim3 numBlocks((sizeX + blockDim-1) / blockDim, (sizeY + blockDim-1) / blockDim);

	//(smesh.nelements + blockSize - 1) / blockSize;
	std::cout << "block size: " << threadsPerBlock.x << " x " << threadsPerBlock.y 
		<< ", grid size: " << numBlocks.x << " x " << numBlocks.y << std::endl;

	// run kernel
	myKernel<<<numBlocks, threadsPerBlock>>>(out, arr, sizeX, sizeY);
	cudaDeviceSynchronize();

	// verify result
	std::vector<T> resultGPU(numEl);
	cudaMemcpy(resultGPU.data(), out, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for(IndexType i = 0; i < numEl; ++i){
		if (resultCPU[i] != resultGPU[i]){
			std::cout << i << ": " << resultCPU[i] << " != " << resultGPU[i] << "\n";
		}
	}
	
}

int main(){
	run_benchmark(500, 500);

	return 0;
}