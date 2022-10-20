#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n) {
        out[tid] = x[tid] + y[tid];
    }
}

int main(){
    int N = 10000000;
    size_t mem_size = sizeof(float) * N;

    float *x, *y, *out;
    float *cuda_x, *cuda_y, *cuda_out;

    // Allocate host CPU memory for x, y
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    out = static_cast<float*>(malloc(mem_size));

    // Initialize x = 1, y = 2
    for(int i = 0; i < N; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Allocate Device CUDA memory for cuda_x and cuda_y, copy them.
    cudaMalloc((void**)&cuda_x, mem_size);
    cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cuda_y, mem_size);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);
    
    // Allocate cuda_out CUDA memory and launch add_kernel
    cudaMalloc((void**)&cuda_out, mem_size);
    int block_size = 256;
    int grid_size = (N + block_size) / block_size;
    add_kernel<<<grid_size, block_size>>>(cuda_x, cuda_y, cuda_out, N);

    // Copy result from GPU into CPU
    cudaMemcpy(out, cuda_out, mem_size, cudaMemcpyDeviceToHost);
    
    // Sync CUDA stream to wait kernel completation
    cudaDeviceSynchronize();

    // Print result and checkout out = 3.
    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    // Free CUDA Memory
    cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_out);

    // Free Host CPU Memory
    free(x);
    free(y);
    free(out);

    return 0;
}