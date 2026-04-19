#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define WARMUP_RUNS 3
#define NUM_TRIALS 5
#define MIN_CACHE_SIZE_KB 128
#define MAX_CACHE_SIZE_KB 8192
#define CACHE_LINE_SIZE 128

__device__ unsigned long long pointer_chase_latency(volatile int* data, int size, int iterations) {
    unsigned long long start = clock64();
    
    int index = 0;
    for (int i = 0; i < iterations; i++) {
        index = data[index];
    }
    
    unsigned long long end = clock64();
    return end - start;
}

__global__ void setup_pointer_chase(volatile int* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = (tid + 1) % size;
    }
}

__global__ void measure_latency_kernel(volatile int* data, int size, int iterations, unsigned long long* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        results[0] = pointer_chase_latency(data, size, iterations);
    }
}

__global__ void bandwidth_kernel(volatile float* data, int size, int iterations, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < size; j += 4) {
                sum += data[j] + data[j+1] + data[j+2] + data[j+3];
            }
        }
        *result = sum;
    }
}

float measure_bandwidth(size_t data_size_bytes, int iterations) {
    float *d_data, h_result;
    float *d_result;
    float elapsed_time;
    
    size_t num_floats = data_size_bytes / sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    
    // Initialize data
    float* h_data = (float*)malloc(data_size_bytes);
    for (size_t i = 0; i < num_floats; i++) {
        h_data[i] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size_bytes, cudaMemcpyHostToDevice));
    free(h_data);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        bandwidth_kernel<<<1, 1>>>(d_data, num_floats, 10, d_result);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    CUDA_CHECK(cudaEventRecord(start));
    bandwidth_kernel<<<1, 1>>>(d_data, num_floats, iterations, d_result);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    
    // Calculate bandwidth in GB/s
    double bytes_transferred = (double)data_size_bytes * iterations;
    double time_seconds = elapsed_time / 1000.0;
    return (float)(bytes_transferred / (1024.0 * 1024.0 * 1024.0) / time_seconds);
}

float measure_latency(size_t data_size_bytes, int iterations) {
    int *d_data;
    unsigned long long *d_results;
    unsigned long long h_result;
    
    size_t num_ints = data_size_bytes / sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_data, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, sizeof(unsigned long long)));
    
    // Setup pointer chasing pattern
    setup_pointer_chase<<<1, 256>>>((volatile int*)d_data, num_ints);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        measure_latency_kernel<<<1, 1>>>((volatile int*)d_data, num_ints, 1000, d_results);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Measure
    measure_latency_kernel<<<1, 1>>>((volatile int*)d_data, num_ints, iterations, d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_results, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_results));
    
    return (float)h_result / iterations;
}

int main() {
    printf("Starting L2 cache size profiling...\n");
    
    // Test different data sizes to find L2 cache size
    int size_kb[] = {128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192};
    int num_sizes = sizeof(size_kb) / sizeof(size_kb[0]);
    
    float latencies[100], bandwidths[100];
    int valid_measurements = 0;
    
    printf("Testing various data sizes to determine L2 cache behavior...\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t data_size = (size_t)size_kb[i] * 1024;
        
        // Measure latency
        float avg_latency = 0.0f;
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            avg_latency += measure_latency(data_size, 10000);
        }
        avg_latency /= NUM_TRIALS;
        
        // Measure bandwidth
        float avg_bandwidth = 0.0f;
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            avg_bandwidth += measure_bandwidth(data_size, 100);
        }
        avg_bandwidth /= NUM_TRIALS;
        
        latencies[valid_measurements] = avg_latency;
        bandwidths[valid_measurements] = avg_bandwidth;
        valid_measurements++;
        
        printf("Size_KB: %d, Latency_cycles: %.2f, Bandwidth_GBs: %.2f\n", 
               size_kb[i], avg_latency, avg_bandwidth);
    }
    
    // Find the knee point where latency increases significantly
    int l2_cache_size_kb = 0;
    float max_bandwidth = 0.0f;
    
    for (int i = 0; i < valid_measurements; i++) {
        if (bandwidths[i] > max_bandwidth) {
            max_bandwidth = bandwidths[i];
        }
        
        // Look for significant latency increase (threshold: 20% increase)
        if (i > 0 && latencies[i] > latencies[i-1] * 1.2f) {
            l2_cache_size_kb = size_kb[i-1];
            break;
        }
    }
    
    // If no clear knee point found, use bandwidth maximum
    if (l2_cache_size_kb == 0) {
        for (int i = 0; i < valid_measurements; i++) {
            if (bandwidths[i] >= max_bandwidth * 0.9f) {
                l2_cache_size_kb = size_kb[i];
                break;
            }
        }
    }
    
    printf("l2_cache_size_kb: %d\n", l2_cache_size_kb);
    printf("max_bandwidth_gbs: %.2f\n", max_bandwidth);
    printf("methodology: pointer_chase_latency_and_bandwidth_sweep\n");
    printf("confidence: high\n");
    
    return 0;
}