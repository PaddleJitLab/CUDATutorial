#include <stdio.h>

__global__ void what_is_my_id_2d(unsigned int * griddim_x, unsigned int * griddim_y,
                         unsigned int * blockdim_x, unsigned int * blockdim_y,
                         unsigned int * calc_thread,
                         unsigned int * blockidx_x, unsigned int * blockidx_y,
                         unsigned int * threadidx_x,  unsigned int * threadidx_y,
                         unsigned int * thread_x, unsigned int * thread_y
                         ){
    
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx ;
    
    griddim_x[thread_idx] = gridDim.x;
    griddim_y[thread_idx] = gridDim.y;
    blockdim_x[thread_idx] = blockDim.x;
    blockdim_y[thread_idx] = blockDim.y;
    calc_thread[thread_idx] = thread_idx;

    blockidx_x[thread_idx] = blockIdx.x;
    blockidx_y[thread_idx] = blockIdx.y;
    threadidx_x[thread_idx] = threadIdx.x;
    threadidx_y[thread_idx] = threadIdx.y;
    thread_x[thread_idx] = idx;
    thread_y[thread_idx] = idy;


}

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16

#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * ARRAY_SIZE_X * ARRAY_SIZE_Y)

unsigned int cpu_griddim_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_griddim_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockdim_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockdim_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockidx_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_blockidx_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_threadidx_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_threadidx_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];


int main(){
    const dim3 num_blocks(1, 4);
    const dim3 num_threads(32, 4);  
    
    unsigned int * gpu_griddim_x;
    unsigned int * gpu_griddim_y;
    unsigned int * gpu_blockdim_x;
    unsigned int * gpu_blockdim_y;
    unsigned int * gpu_calc_thread;
    unsigned int * gpu_blockidx_x;
    unsigned int * gpu_blockidx_y;
    unsigned int * gpu_threadidx_x;
    unsigned int * gpu_threadidx_y;
    unsigned int * gpu_thread_x;
    unsigned int * gpu_thread_y;

    
    cudaMalloc((void**)&gpu_griddim_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_griddim_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockdim_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockdim_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockidx_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_blockidx_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_threadidx_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_threadidx_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread_y, ARRAY_SIZE_IN_BYTES);

    
    what_is_my_id_2d<<<num_blocks, num_threads>>>(gpu_griddim_x, gpu_griddim_y, gpu_blockdim_x, gpu_blockdim_y, gpu_calc_thread, gpu_blockidx_x, gpu_blockidx_y, gpu_threadidx_x, gpu_threadidx_y, gpu_thread_x, gpu_thread_y);

    cudaMemcpy(cpu_griddim_x, gpu_griddim_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_griddim_y, gpu_griddim_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockdim_x, gpu_blockdim_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockdim_y, gpu_blockdim_y,ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread,ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockidx_x, gpu_blockidx_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_blockidx_y, gpu_blockidx_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_threadidx_x, gpu_threadidx_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_threadidx_y, gpu_threadidx_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread_x, gpu_thread_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread_y, gpu_thread_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
   
    cudaFree(gpu_griddim_x);
    cudaFree(gpu_griddim_y);
    cudaFree(gpu_blockdim_x);
    cudaFree(gpu_blockdim_y);
    cudaFree(gpu_calc_thread);
    cudaFree(gpu_blockidx_x);
    cudaFree(gpu_blockidx_y);
    cudaFree(gpu_threadidx_x);
    cudaFree(gpu_threadidx_y);
    cudaFree(gpu_thread_x);
    cudaFree(gpu_thread_y);


    for(int y=0; y < ARRAY_SIZE_Y; y++){
        for(int x=0; x < ARRAY_SIZE_X; x++){
        printf("graddim_x %2u - graddim_y %2u - blockdim_x %2u - blockdim_y %2u -cac_thread %3u - blockidx_x %2u -  blockidx_y %2u- threadidx_x %2u -  threadidx_y %2u - thread_x %2u - thread_y %2u \n",
         cpu_griddim_x[y][x], cpu_griddim_y[y][x], cpu_blockdim_x[y][x], cpu_blockdim_y[y][x], cpu_calc_thread[y][x], cpu_blockidx_x[y][x], cpu_blockidx_y[y][x], cpu_threadidx_x[y][x], cpu_threadidx_y[y][x], cpu_thread_x[y][x], cpu_thread_y[y][x]);
        }
    }

}
