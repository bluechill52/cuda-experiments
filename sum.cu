#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>


#define N (1 << 20)
#define TILE_SIZE 32

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        atomicAdd(result, A[idx]);
    }
}

__global__ void sum_reduction_optimized(const float* A, float* result, int n) {
    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    tile[tid] = (idx < n) ? A[idx] : 0.0f;   // bounds-checked load too
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tile[tid] += tile[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, tile[0]);   // ONE atomic per block now
    }
}

template <int BLOCK_SIZE>
__global__ void reduce_kernel(const float* __restrict__ A, float* __restrict__ out, int n) {
    __shared__ float tile[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int gridStride = BLOCK_SIZE * gridDim.x;

    float sum = 0.0f;

    // --- Vectorized float4 loads: 4 elements per memory transaction ---
    int n4 = n / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    for (int i = idx; i < n4; i += gridStride) {
        float4 v = A4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // tail elements when n isn't divisible by 4
    for (int i = n4 * 4 + idx; i < n; i += gridStride) {
        sum += A[i];
    }

    tile[tid] = sum;
    __syncthreads();

    // --- Tree reduction, fully unrolled at compile time (BLOCK_SIZE is
    //     a template param, so these "if"s are dead-code eliminated —
    //     no runtime loop, no runtime branch) ---
    if (BLOCK_SIZE >= 1024) { if (tid < 512) tile[tid] += tile[tid + 512]; __syncthreads(); }
    if (BLOCK_SIZE >=  512) { if (tid < 256) tile[tid] += tile[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >=  256) { if (tid < 128) tile[tid] += tile[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >=  128) { if (tid <  64) tile[tid] += tile[tid +  64]; __syncthreads(); }

    // --- Last warp: shuffle-based finish, no shared mem, no barriers ---
    if (tid < 32) {
        float val = tile[tid] + tile[tid + 32];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        // --- One plain store per block, no atomics anywhere ---
        if (tid == 0) out[blockIdx.x] = val;
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

template <typename T>
T run_sum_reduction_v3(int offset, float& kernelTimeMs) {
    T *A = (T*) malloc(sizeof(T) * N);
    for (int i = 0; i < N; i++) A[i] = i + offset;

    T *dA, *dBlockSums, *dResult;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(T) * N));
    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(T) * N, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;

    // Size the grid to what this specific GPU can actually run concurrently,
    // not to N — avoids both over- and under-subscribing the SMs.
    int maxActiveBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, reduce_kernel<threadsPerBlock>,
        threadsPerBlock, threadsPerBlock * sizeof(T)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int numBlocks = std::min(maxActiveBlocksPerSM * prop.multiProcessorCount,
                              (N + threadsPerBlock - 1) / threadsPerBlock);

    CUDA_CHECK(cudaMalloc(&dBlockSums, sizeof(T) * numBlocks));
    CUDA_CHECK(cudaMalloc(&dResult, sizeof(T)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up (not timed)
    reduce_kernel<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(dA, dBlockSums, N);
    reduce_kernel<threadsPerBlock><<<1, threadsPerBlock>>>(dBlockSums, dResult, numBlocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    reduce_kernel<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(dA, dBlockSums, N);       // N -> numBlocks
    reduce_kernel<threadsPerBlock><<<1, threadsPerBlock>>>(dBlockSums, dResult, numBlocks);  // numBlocks -> 1
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&kernelTimeMs, start, stop));

    T result;
    CUDA_CHECK(cudaMemcpy(&result, dResult, sizeof(T), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dBlockSums); cudaFree(dResult);
    free(A);

    return result;
}

template <typename T>
T run_sum_reduction_optimized(int offset, float& kernelTimeMs) {
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
    sum_reduction_optimized<<<blocksPerGrid, threadsPerBlock, threadsPerBlock>>>(dA, dResult, N);
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

    // Test correctness
    float reference = 0.0;
    for(int i=0;i<N;i++) {
        reference += i + offset;
    }

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

    float relErr = fabs(reference - result) / reference;
    std::cout << "Un optimized method" << std::endl;
    std::cout << "Relative error b/w reference and computed " << relErr << std::endl;
    std::cout << "Min kernel time - " << minKernelTimeMs << std::endl;
    std::cout << "Mean kernel time - " << meanKernelTimeMs << std::endl;

    for(int i=0;i<num_trials;i++) {
        result = run_sum_reduction_v3<float>(offset, kernelTimeMs);
        minKernelTimeMs = min(minKernelTimeMs, kernelTimeMs);
        meanKernelTimeMs += kernelTimeMs;
    }

    meanKernelTimeMs /= num_trials;

    relErr = fabs(reference - result) / reference;
    std::cout << "Optimized method" << std::endl;
    std::cout << "Relative error b/w reference and computed " << relErr << std::endl;
    std::cout << "Min kernel time - " << minKernelTimeMs << std::endl;
    std::cout << "Mean kernel time - " << meanKernelTimeMs << std::endl;
    return 0;
}
