#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_TRIALS 20
#define WARMUP_TRIALS 3
#define ARRAY_SIZE (1 << 22)  // 4M elements
#define THREADS_PER_BLOCK 256

// Kernel: vector addition (compute + memory bound mix)
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float sum = 0.0f;
        // Unroll a bit to increase arithmetic intensity
        for (int j = 0; j < 8; j++) {
            if (i + j * N < N * 8) {
                int idx2 = (i + j * 1024) % N;
                sum += A[idx2] * B[idx2];
            }
        }
        C[i] = sum;
    }
}

// Kernel: pure compute (FLOPS-focused)
__global__ void computeKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float x = data[i];
        // Series of FMA operations
        for (int j = 0; j < 100; j++) {
            x = x * 1.0001f + 0.0001f;
            x = x * x + 1.0f;
            x = sqrtf(x);
            x = sinf(x);
        }
        data[i] = x;
    }
}

// Kernel: memory bandwidth focused
__global__ void memCopyKernel(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        dst[i] = src[i] * 2.0f + 1.0f;
    }
}

// Kernel: mixed workload representative of typical GPU tasks
__global__ void mixedKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float a = A[idx];
    float b = B[idx];
    float result = 0.0f;

    // Compute phase
    for (int i = 0; i < 50; i++) {
        result += a * b;
        result = sinf(result) + cosf(result);
        a = result * 0.999f + b;
    }
    // Memory phase
    C[idx] = result;
    C[idx + (N % 2)] += result;  // extra write
}

int main() {
    cudaError_t err;
    int N = ARRAY_SIZE;
    int bytes = N * sizeof(float);
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc B failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc C failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Initialize host data
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ---- Benchmark mixed kernel ----
    double elapsedTimes[NUM_TRIALS];
    unsigned long long cycleTimes[NUM_TRIALS];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int w = 0; w < WARMUP_TRIALS; w++) {
        mixedKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

    // Timed trials
    for (int t = 0; t < NUM_TRIALS; t++) {
        // Cycle timing
        unsigned long long startCycle = clock64();
        cudaEventRecord(start);

        mixedKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        unsigned long long endCycle = clock64();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        elapsedTimes[t] = ms;
        cycleTimes[t] = (endCycle - startCycle);
    }

    // ---- Benchmark compute kernel ----
    double computeTimes[NUM_TRIALS];
    unsigned long long computeCycles[NUM_TRIALS];

    // Warmup
    for (int w = 0; w < WARMUP_TRIALS; w++) {
        computeKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, N);
    }
    cudaDeviceSynchronize();

    for (int t = 0; t < NUM_TRIALS; t++) {
        unsigned long long startCycle = clock64();
        cudaEventRecord(start);

        computeKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        unsigned long long endCycle = clock64();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        computeTimes[t] = ms;
        computeCycles[t] = (endCycle - startCycle);
    }

    // ---- Benchmark memory kernel ----
    double memTimes[NUM_TRIALS];
    unsigned long long memCycles[NUM_TRIALS];

    // Warmup
    for (int w = 0; w < WARMUP_TRIALS; w++) {
        memCopyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_C, N);
    }
    cudaDeviceSynchronize();

    for (int t = 0; t < NUM_TRIALS; t++) {
        unsigned long long startCycle = clock64();
        cudaEventRecord(start);

        memCopyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        unsigned long long endCycle = clock64();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        memTimes[t] = ms;
        memCycles[t] = (endCycle - startCycle);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute medians
    // Sort helper
    for (int t = 0; t < NUM_TRIALS - 1; t++) {
        for (int s = t + 1; s < NUM_TRIALS; s++) {
            if (elapsedTimes[s] < elapsedTimes[t]) {
                double tmp = elapsedTimes[t];
                elapsedTimes[t] = elapsedTimes[s];
                elapsedTimes[s] = tmp;
                unsigned long long tmp2 = cycleTimes[t];
                cycleTimes[t] = cycleTimes[s];
                cycleTimes[s] = tmp2;
            }
        }
    }

    double medianMixed = elapsedTimes[NUM_TRIALS / 2];
    unsigned long long medianCyclesMixed = cycleTimes[NUM_TRIALS / 2];

    // Sort compute
    for (int t = 0; t < NUM_TRIALS - 1; t++) {
        for (int s = t + 1; s < NUM_TRIALS; s++) {
            if (computeTimes[s] < computeTimes[t]) {
                double tmp = computeTimes[t];
                computeTimes[t] = computeTimes[s];
                computeTimes[s] = tmp;
                unsigned long long tmp2 = computeCycles[t];
                computeCycles[t] = computeCycles[s];
                computeCycles[s] = tmp2;
            }
        }
    }

    double medianCompute = computeTimes[NUM_TRIALS / 2];
    unsigned long long medianCyclesCompute = computeCycles[NUM_TRIALS / 2];

    // Sort memory
    for (int t = 0; t < NUM_TRIALS - 1; t++) {
        for (int s = t + 1; s < NUM_TRIALS; s++) {
            if (memTimes[s] < memTimes[t]) {
                double tmp = memTimes[t];
                memTimes[t] = memTimes[s];
                memTimes[s] = tmp;
                unsigned long long tmp2 = memCycles[t];
                memCycles[t] = memCycles[s];
                memCycles[s] = tmp2;
            }
        }
    }

    double medianMemory = memTimes[NUM_TRIALS / 2];
    unsigned long long medianCyclesMemory = memCycles[NUM_TRIALS / 2];

    // Compute GFLOPS for compute kernel
    // Each element: 100 iterations * 4 FMA-like ops + 2 transcendental = ~500 flops
    double totalFlops = (double)N * 500.0;
    double gflopsCompute = totalFlops / (medianCompute * 1e6);

    // Compute effective bandwidth for memory kernel
    // Read 4 bytes per element, write 4 bytes per element = 8 bytes/element
    double totalBytes = (double)N * 8.0;
    double bandwidthGBps = totalBytes / (medianMemory * 1e6);

    // Composite score: harmonic mean of normalized metrics
    // Use a balanced composite metric
    double compositeScore = (gflopsCompute * 0.4) + (bandwidthGBps * 0.3) + ((1000.0 / medianMixed) * 0.3);

    // Output parseable results
    printf("unknown: %.4f\n", compositeScore);
    printf("median_mixed_ms: %.4f\n", medianMixed);
    printf("median_compute_ms: %.4f\n", medianCompute);
    printf("median_memory_ms: %.4f\n", medianMemory);
    printf("gflops_compute: %.2f\n", gflopsCompute);
    printf("bandwidth_gb_s: %.2f\n", bandwidthGBps);
    printf("num_trials: %d\n", NUM_TRIALS);
    printf("array_size: %d\n", N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
