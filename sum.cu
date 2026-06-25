#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


// #define N (1 << 20)
#define N 200


__global__ void sum_reduction(const float* A, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        atomicAdd(result, A[i]);
    }
}


template <typename T>
T run_sum_reduction(int offset) {
    T *A = (T*) malloc(sizeof(T) * N);
    for(int i=0;i<N;i++) {
        A[i] = i + offset;
    }

    T* dA;
    T* dResult;

    cudaMalloc(&dA, sizeof(T) * N);
    cudaMalloc(&dResult, sizeof(T));

    cudaMemcpy(dA, A, sizeof(T) * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sum_reduction<<<blocksPerGrid, threadsPerBlock>>>(dA, dResult, N);

    T result;
    cudaMemcpy(&result, dResult, sizeof(T), cudaMemcpyDeviceToHost);

    return result;
}

int main() {
    int offset = 1;
    float result = run_sum_reduction<float>(1);

    // Test correctness
    float reference = 0.0;
    for(int i=0;i<N;i++) {
        reference += i + offset;
    }

    std::cout << "Reference value - " << reference << std::endl;
    std::cout << "Computed value - " << result << std::endl;
    return 0;
}
