#include <stdio.h>

__global__ void what_is_my_id_2d(unsigned int * block_x, unsigned int * block_y,
                         unsigned int * threadid_x,  unsigned int * threadid_y,
                         unsigned int * thread_x, unsigned int * thread_y, 
                         unsigned int * griddim_x, unsigned int * griddim_y,
                         unsigned int * blockdim_x, unsigned int * blockdim_y,
                         unsigned int * calc_thread){
    
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx ;

    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    threadid_x[thread_idx] = threadIdx.x;
    threadid_y[thread_idx] = threadIdx.y;
    thread_x[thread_idx] = idx;
    thread_y[thread_idx] = idy;
    griddim_x[thread_idx] = gridDim.x;
    griddim_y[thread_idx] = gridDim.y;
    blockdim_x[thread_idx] = blockDim.x;
    blockdim_y[thread_idx] = blockDim.y;

    calc_thread[thread_idx] = thread_idx;
}

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16

#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * ARRAY_SIZE_X * ARRAY_SIZE_Y)

unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_threadid_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_threadid_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_griddim_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_griddim_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockdim_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockdim_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(){
    const dim3 num_blocks(1, 4);
    const dim3 num_threads(32, 4);  
    
    unsigned int * gpu_block_x;
    unsigned int * gpu_block_y;
    unsigned int * gpu_threadid_x;
    unsigned int * gpu_threadid_y;
    unsigned int * gpu_thread_x;
    unsigned int * gpu_thread_y;
    unsigned int * gpu_griddim_x;
    unsigned int * gpu_griddim_y;
    unsigned int * gpu_blockdim_x;
    unsigned int * gpu_blockdim_y;
    unsigned int * gpu_calc_thread;
    
    cudaMalloc((void**)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_threadid_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_threadid_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_griddim_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_griddim_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockdim_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockdim_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    
    what_is_my_id_2d<<<num_blocks, num_threads>>>(gpu_block_x, gpu_block_y, gpu_threadid_x, gpu_threadid_y, gpu_thread_x, gpu_thread_y, gpu_griddim_x, gpu_griddim_y, gpu_blockdim_x, gpu_blockdim_y, gpu_calc_thread);

    cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_threadid_x, gpu_threadid_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_threadid_y, gpu_threadid_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread_x, gpu_thread_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread_y, gpu_thread_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_griddim_x, gpu_griddim_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_griddim_y, gpu_griddim_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockdim_x, gpu_blockdim_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockdim_y, gpu_blockdim_y,ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread,ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


    cudaFree(gpu_block_x);
    cudaFree(gpu_block_y);
    cudaFree(gpu_threadid_x);
    cudaFree(gpu_threadid_y);
    cudaFree(gpu_thread_x);
    cudaFree(gpu_thread_y);
    cudaFree(gpu_griddim_x);
    cudaFree(gpu_griddim_y);
    cudaFree(gpu_blockdim_x);
    cudaFree(gpu_blockdim_y);
    cudaFree(gpu_calc_thread);

    for(int y=0; y < ARRAY_SIZE_Y; y++){
        for(int x=0; x < ARRAY_SIZE_X; x++){
        printf("graddim_x %2u - graddim_y %2u - blockdim_x %2u - blockdim_y %2u -cac_thread %3u - block_x %2u -  block_y %2u- threadid_x %2u -  threadid_y %2u - thread_x %2u - thread_y %2u \n",
         cpu_griddim_x[y][x], cpu_griddim_y[y][x], cpu_blockdim_x[y][x], cpu_blockdim_y[y][x], cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x], cpu_threadid_x[y][x], cpu_threadid_y[y][x], cpu_thread_x[y][x], cpu_thread_x[y][x]);
        }
    }

}
