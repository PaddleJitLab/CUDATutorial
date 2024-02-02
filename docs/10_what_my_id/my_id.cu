#include <stdio.h>

template<unsigned int warpsize>
__global__ void what_is_my_id(unsigned int * block,
                         unsigned int * thread, 
                         unsigned int * warp, 
                         unsigned int * calc_thread ){
    const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;

    warp[thread_idx] = threadIdx.x / warpsize;
    calc_thread[thread_idx] = thread_idx;
}

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * ARRAY_SIZE)

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main(){
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = 64;
    const unsigned int warp_size = 32;

    unsigned int * gpu_block;
    unsigned int * gpu_thread;
    unsigned int * gpu_warp;
    unsigned int * gpu_calc_thread;

    unsigned int i;
    
    cudaMalloc((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

    what_is_my_id<warp_size><<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    for(i=0; i< ARRAY_SIZE; i++){
        printf("cac_thread %3u - block %2u - warp %3u - thread %2u\n", cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }

}
