#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Simple matrix multiplication kernel
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * Cvalue + beta * C[row * N + col];
    }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Warmup
    sgemm_tiled<<<numBlocks, threadsPerBlock>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    // Actual run
    sgemm_tiled<<<numBlocks, threadsPerBlock>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
