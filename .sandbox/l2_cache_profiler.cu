#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Kernel to measure memory access performance with different working set sizes
__global__ void memory_access_kernel(float* data, int* indices, int num_accesses, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    // Each thread performs multiple random accesses
    for (int i = tid; i < num_accesses; i += stride) {
        int idx = indices[i % (num_accesses / stride + 1)];
        sum += data[idx];
    }
    
    // Prevent compiler optimization
    if (sum != 0.0f) {
        result[tid] = sum;
    }
}

// Function to generate random access pattern
void generate_random_indices(int* indices, int size, int max_index) {
    for (int i = 0; i < size; i++) {
        indices[i] = rand() % max_index;
    }
}

int main() {
    // Test different working set sizes (in bytes)
    size_t sizes[] = {
        1024,           // 1KB
        4096,           // 4KB
        16384,          // 16KB
        65536,          // 64KB
        262144,         // 256KB
        1048576,        // 1MB
        4194304,        // 4MB
        16777216,       // 16MB
        67108864,       // 64MB
        268435456       // 256MB
    };
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Device pointers
    float *d_data, *d_result;
    int *d_indices;
    
    // Host pointers
    float *h_result;
    int *h_indices;
    
    // Number of memory accesses per test
    const int num_accesses = 1000000;
    
    printf("Working Set Size (KB)\tTime (ms)\tBandwidth (GB/s)\n");
    printf("==================================================\n");
    
    srand(time(NULL));
    
    for (int i = 0; i < num_sizes; i++) {
        size_t data_size = sizes[i];
        int num_elements = data_size / sizeof(float);
        
        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_data, data_size));
        CUDA_CHECK(cudaMalloc(&d_result, 256 * sizeof(float))); // Assume max 256 threads
        CUDA_CHECK(cudaMalloc(&d_indices, num_accesses * sizeof(int)));
        
        h_result = (float*)malloc(256 * sizeof(float));
        h_indices = (int*)malloc(num_accesses * sizeof(int));
        
        // Initialize data
        CUDA_CHECK(cudaMemset(d_data, 1, data_size));
        generate_random_indices(h_indices, num_accesses, num_elements);
        CUDA_CHECK(cudaMemcpy(d_indices, h_indices, num_accesses * sizeof(int), cudaMemcpyHostToDevice));
        
        // Configure kernel launch parameters
        int block_size = 256;
        int grid_size = min(256, (num_accesses + block_size - 1) / block_size);
        
        // Warm up
        memory_access_kernel<<<grid_size, block_size>>>(d_data, d_indices, num_accesses, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Measure performance
        CUDA_CHECK(cudaEventRecord(start));
        memory_access_kernel<<<grid_size, block_size>>>(d_data, d_indices, num_accesses, d_result);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Calculate bandwidth
        double bandwidth = (double)num_accesses * sizeof(float) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        
        printf("%.1f\t\t\t%.3f\t\t%.2f\n", 
               (double)data_size / 1024.0, 
               milliseconds, 
               bandwidth);
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_result));
        CUDA_CHECK(cudaFree(d_indices));
        free(h_result);
        free(h_indices);
    }
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\nNote: Look for the performance drop point to estimate L2 cache size.\n");
    printf("The L2 cache size is typically where bandwidth decreases significantly.\n");
    
    return 0;
}