#include <vector>
#include <cstdint>
#include <iostream>
#include <memory>
#include <chrono>

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

using FloatType = float;

__device__ __host__ IndexType flatIdx(IndexType x, IndexType y, IndexType sizeX, IndexType sizeY) {
	if constexpr (COLUMN_MAJOR){
		return x + y * sizeX;
	} else {
		return x * sizeY + y;
	} 
}

__global__ void naiveKernel(RESTRICT FloatType* out, RESTRICT const FloatType* arr, IndexType sizeX, IndexType sizeY){
	const IndexType i = (blockIdx.x * blockDim.x) + threadIdx.x;
	const IndexType j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i > sizeX || j > sizeY){
		return;
	}

	for (IndexType k = 0; k < sizeX; ++k){
		out[flatIdx(k, i, sizeX, sizeY)] += arr[flatIdx(i, j, sizeX, sizeY)];
	}
}

__global__ void atomicsKernel(RESTRICT FloatType* out, RESTRICT const FloatType* arr, IndexType sizeX, IndexType sizeY){
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

// @return num errors, mean l1 error
std::pair<int, FloatType> compareResults(const FloatType* arr0, const FloatType* arr1, IndexType size) {
	int numErrors = 0;
	FloatType totalL1Error = 0.0;
	for(IndexType i = 0; i < size; ++i){
		const FloatType err = std::abs(arr0[i] - arr1[i]);
		if (err){
			++numErrors;
			totalL1Error += err;
			//std::cout << i << ": " << resultCPU[i] << " != " << resultGPU[i] << "\n";
		}
	}

	return {numErrors, totalL1Error / size};
}

template<typename Kernel>
void measureKernel(const std::string& name, Kernel kernel, FloatType* arr, FloatType* out, const FloatType* hostInit, const FloatType* hostReference, IndexType sizeX, IndexType sizeY, int repeats = 512){
	std::cout << name << "\n";
	// determine block size
	int maxBlockSize = 0;
    int minGridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, kernel, 0, 0);
	const unsigned blockDim = std::sqrt(maxBlockSize);
	const dim3 threadsPerBlock(blockDim, blockDim);
	const dim3 numBlocks((sizeX + blockDim-1) / blockDim, (sizeY + blockDim-1) / blockDim);

	std::cout << "block size: " << threadsPerBlock.x << " x " << threadsPerBlock.y 
		<< ", grid size: " << numBlocks.x << " x " << numBlocks.y << std::endl;

	const IndexType numEl = sizeX * sizeY;
	const size_t bytes = numEl * sizeof(FloatType);
	cudaMemcpy(arr, hostInit, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemset(out, 0, bytes);

	// verify correctness
	kernel<<<numBlocks, threadsPerBlock>>>(out, arr, sizeX, sizeY);
	std::vector<FloatType> resultGPU(numEl);
	cudaMemcpy(resultGPU.data(), out, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	auto [numErrors, meanL1Error] = compareResults(hostReference, resultGPU.data(), sizeX * sizeY);
	std::cout << "num errors: " << numErrors << ", mean L1: " << meanL1Error << "\n";

	// run kernel
	double totalTime = 0.0;
	double minTime = std::numeric_limits<FloatType>::max();
	double maxTime = 0.0;
	for(int i = 0; i < repeats; ++i){
		auto start = std::chrono::high_resolution_clock::now();
		kernel<<<numBlocks, threadsPerBlock>>>(out, arr, sizeX, sizeY);
		cudaDeviceSynchronize();
		auto stop = std::chrono::high_resolution_clock::now();
		const double t = std::chrono::duration<double>(stop-start).count();
		totalTime += t;
		if (t < minTime) minTime = t;
		if (t > maxTime) maxTime = t;
	}

	std::cout << "total: " << totalTime << "s\n" 
		<< "mean: " << totalTime / repeats << "\n"
		<< "min: " << minTime << "\n"
		<< "max: " << maxTime << "\n\n";
}

//template<typename T> 
void run_benchmark(IndexType sizeX, IndexType sizeY) {
	const IndexType numEl = sizeX * sizeY;
	// prepare input data
	std::vector<FloatType> hostBuf(numEl, 0.0);
	auto setValue = [&](IndexType beginX, IndexType endX, IndexType beginY, IndexType endY, FloatType value){
		for(IndexType iy = beginY; iy < endY; ++iy){
			for(IndexType ix = beginX; ix < endX; ++ix){
				hostBuf[flatIdx(ix, iy, sizeX, sizeY)] = value;
			}
		}
	};
	setValue(100, 200, 100, 200, 1.0);
	setValue(350, 450, 350, 450, 1.0);

	std::vector<FloatType> resultCPU(numEl, 0.0);
	for(IndexType iy = 0; iy < sizeY; ++iy){
		for(IndexType ix = 0; ix < sizeX; ++ix){
			for(IndexType ik = 0; ik < sizeX; ++ik){
				resultCPU[flatIdx(ik, ix, sizeX, sizeY)] += hostBuf[flatIdx(ix, iy, sizeX, sizeY)];
			}
		}
	}

	// allocate GPU buffers
	const size_t bytes = hostBuf.size() * sizeof(FloatType);
	FloatType* arr = nullptr;
	FloatType* out = nullptr;
	cudaMalloc(&arr, bytes);
	cudaMalloc(&out, bytes);

	measureKernel("cuda naive", naiveKernel, arr, out, hostBuf.data(), resultCPU.data(), sizeX, sizeY);
	measureKernel("cuda atomics", atomicsKernel, arr, out, hostBuf.data(), resultCPU.data(), sizeX, sizeY);
	// naiveKernel
	
	cudaFree(arr);
	cudaFree(out);
}

int main(){
	run_benchmark(500, 500);

	return 0;
}