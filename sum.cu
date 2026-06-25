#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>


#define N (1 << 20)


__global__ void sum_reduction(const float* A, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) {
        // atomicAdd(result, A[i]);
        *result += A[i];
    }
}

int main() {
    float *A = (float*) malloc(sizeof(float) * N);
    for(int i=0;i<N;i++) {
        A[i] = i + 1;
    }

    float* dA;
    float* dResult;

    cudaMalloc(&dA, sizeof(float) * N);
    cudaMalloc(&dResult, sizeof(float));

    cudaMemcpy(dA, A, sizeof(float) * N, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sum_reduction<<<blocksPerGrid, threadsPerBlock>>>(dA, dResult, N);

    float result;
    cudaMemcpy(&result, dResult, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << result << std::endl;
}