#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================
// GPU 性能基准测试 - 通过实际测量计算 sm__throughput 
// 和 gpu__compute_memory_throughput 指标
// ============================================================

// 计算密集型 kernel - 用于测量 SM 吞吐量
__global__ void compute_benchmark_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ result,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    double acc = 0.0;
    const int iterations = 10000000;
    
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma unroll 1
        for (int k = 0; k < 8; ++k) {
            const int i = (idx + k * 37 + iter * 13) % n;
            volatile double va = a[i];
            volatile double vb = b[i];
            // FMA 操作最大化 SM 利用率
            acc = va * vb + acc;
        }
    }
    
    result[idx % n] = acc;
}

// 内存带宽密集型 kernel - 用于测量内存吞吐量
__global__ void memory_benchmark_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += total_threads) {
        volatile double val = input[i];
        output[i] = val * 2.0 + 1.0;
    }
}

int main() {
    // 查询 SM 数量
    int sm_count;
    cudaError_t err = cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to query SM count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // 使用 sm_count * 4 blocks x 256 threads
    const int blocks = sm_count * 4;
    const int threads = 256;
    
    // 分配 128MB 内存以绕过 L2 缓存
    const size_t n = 16777216; // 16M doubles = 128MB
    const size_t bytes = n * sizeof(double);
    
    double *h_a, *h_b, *h_result;
    double *d_a, *d_b, *d_result;
    
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_result, bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, bytes);
    
    // 初始化数据
    for (size_t i = 0; i < n; ++i) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = (double)rand() / RAND_MAX;
    }
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Warmup
    compute_benchmark_kernel<<<blocks, threads>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();
    
    memory_benchmark_kernel<<<blocks, threads>>>(d_a, d_result, n);
    cudaDeviceSynchronize();
    
    // 运行计算基准测试（不测量，让 ncu 来测量）
    compute_benchmark_kernel<<<blocks, threads>>>(d_a, d_b, d_result, n);
    cudaDeviceSynchronize();
    
    // 运行内存基准测试（不测量，让 ncu 来测量）
    memory_benchmark_kernel<<<blocks, threads>>>(d_a, d_result, n);
    cudaDeviceSynchronize();
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_result);
    
    cudaDeviceReset();
    return 0;
}
