#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


#define N (1 << 20)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__global__ void sum_reduction(const float* A, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        atomicAdd(result, A[i]);
    }
}


template <typename T>
T run_sum_reduction(int offset, float& kernelTimeMs) {
    T *A = (T*) malloc(sizeof(T) * N);
    for(int i=0;i<N;i++) {
        A[i] = i + offset;
    }

    T* dA;
    T* dResult;

    CUDA_CHECK(cudaMalloc(&dA, sizeof(T) * N));
    CUDA_CHECK(cudaMalloc(&dResult, sizeof(T)));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(T) * N, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch (not timed) — first launch pays for context/JIT overhead
    sum_reduction<<<blocksPerGrid, threadsPerBlock>>>(dA, dResult, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(dResult, 0, sizeof(T)));  // reset accumulator after warm-up

    CUDA_CHECK(cudaEventRecord(start));
    sum_reduction<<<blocksPerGrid, threadsPerBlock>>>(dA, dResult, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());  // catch launch errors
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));

    T result;
    CUDA_CHECK(cudaMemcpy(&result, dResult, sizeof(T), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dResult);
    free(A);

    return result;
}

int main() {
    int offset = 1;
    const int num_trials = 100;

    float kernelTimeMs;
    float minKernelTimeMs;
    float meanKernelTimeMs;
    float result;
    for(int i=0;i<num_trials;i++) {
        result = run_sum_reduction<float>(offset, kernelTimeMs);
        minKernelTimeMs = min(minKernelTimeMs, kernelTimeMs);
        meanKernelTimeMs += kernelTimeMs;
    }

    meanKernelTimeMs /= num_trials;

    // Test correctness
    float reference = 0.0;
    for(int i=0;i<N;i++) {
        reference += i + offset;
    }


    std::cout << "Reference value - " << reference << std::endl;
    std::cout << "Computed value - " << result << std::endl;
    std::cout << "Min kernel time - " << minKernelTimeMs << std::endl;
    std::cout << "Mean kernel time - " << meanKernelTimeMs << std::endl;
    return 0;
}
