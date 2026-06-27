#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Assume
// Input A - M x K,
// Input B - K x N,
// Output C - M x N
__global__ void gemm_naive(float* A, float* B, float* C, int M, int K, int N) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        for(size_t k{0};k<K;k++) {
            C[row * N + col] += A[row * K + k] * B[k * N + col];
        }
    }
}

void print_matrix(float* matrix, int num_rows, int num_cols) {
    for(int i=0;i<num_rows;i++) {
        for(int j=0;j<num_cols;j++) {
            std::cout << matrix[i * num_cols + j] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

bool verify(float* A, float* B, float* C, int M, int K, int N) {
    const float relTol = 1e-3f;
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            double tmp = 0.0;
            for(int k=0;k<K;k++) {
                tmp += (double) A[i * K + k] * (double) B[k * N +j];
            }

            // tmp is the reference value - it should match with computed
            // value at the same location from cuda
            double relErr = fabs(tmp - (double) C[i * N + j]) / fmax(fabs(tmp), 1e-6);
            if(relErr > relTol) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    int M = 1024;
    int K = 1024;
    int N = 1024;

    size_t sA = sizeof(float) * M * K;
    size_t sB = sizeof(float) * K * N;
    size_t sC = sizeof(float) * M * N;

    float* A = (float*) malloc(sA);
    float* B = (float*) malloc(sB);
    float* C = (float*) malloc(sC);

    for(int i=0;i<M;i++) {
        for(int j=0;j<K;j++) {
            A[i * K + j] = i * K + j;
        }
    }

    for(int i=0;i<K;i++) {
        for(int j=0;j<N;j++) {
            B[i * N + j] = i * N + j + 1;
        }
    }

    /*
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            for(int k=0;k<K;k++) {
                C[i * N + j] += A[i * K + k] * B[k * N +j];
            }
        }
    }

    print_matrix(A, M, K);
    print_matrix(B, K, N);
    print_matrix(C, M, N);
    */

    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(&dA, sA));
    CUDA_CHECK(cudaMalloc(&dB, sB));
    CUDA_CHECK(cudaMalloc(&dC, sC));

    CUDA_CHECK(cudaMemcpy(dA, A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sB, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.y, (N + blockDim.y - 1) / blockDim.y);

    gemm_naive<<<gridDim, blockDim>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(C, dC, sC, cudaMemcpyDeviceToHost));

    std::cout << (verify(A, B, C, M, K, N) ? \
                "CUDA computed matrix matches with reference" : \
                "Mismatch between CUDA computed matrix and reference") << std::endl;
}