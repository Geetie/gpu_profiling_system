#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void unknown_kernel(volatile int* dummy, int iterations) {
    unsigned long long start_clock = clock64();
    
    // Unknown operation - basic computation loop
    int sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += i * 31;
        sum = sum ^ (sum >> 1);
    }
    
    unsigned long long end_clock = clock64();
    
    // Prevent compiler optimization
    *dummy = sum;
    
    // Output cycle count for this thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("cycles: %llu\n", end_clock - start_clock);
    }
}

int main() {
    const int num_trials = 10;
    const int iterations = 1000000;
    
    float* h_times = (float*)malloc(num_trials * sizeof(float));
    unsigned long long* h_cycles = (unsigned long long*)malloc(num_trials * sizeof(unsigned long long));
    
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start_event);
        
        unknown_kernel<<<1, 1>>>(d_dummy, iterations);
        
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        h_times[trial] = elapsed_ms;
    }
    
    // Read back the dummy value to ensure kernel executed
    int h_dummy;
    cudaMemcpy(&h_dummy, d_dummy, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate median wall-clock time
    for (int i = 0; i < num_trials - 1; i++) {
        for (int j = i + 1; j < num_trials; j++) {
            if (h_times[i] > h_times[j]) {
                float temp = h_times[i];
                h_times[i] = h_times[j];
                h_times[j] = temp;
            }
        }
    }
    
    float median_time_ms = h_times[num_trials / 2];
    
    printf("unknown: %.6f\n", median_time_ms);
    printf("confidence: 0.8\n");
    printf("method_used: custom micro-benchmark\n");
    printf("num_trials: %d\n", num_trials);
    
    free(h_times);
    free(h_cycles);
    cudaFree(d_dummy);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    return 0;
}