#include <stdio.h>

__global__ void cuda_say_hello(){
    printf("Hello world, CUDA! %d\n", threadIdx.x);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<1,1>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}