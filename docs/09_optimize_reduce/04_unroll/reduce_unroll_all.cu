#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int len = 32 * 1024 * 1024;

template <unsigned int BLOCKSIZE>
__device__ void warp_reduce(volatile int *sdata, int tid)
{
    if (BLOCKSIZE >= 64)
    {
        sdata[tid] += sdata[tid + 32];
    }
    if (BLOCKSIZE >= 32)
    {
        sdata[tid] += sdata[tid + 16];
    }
    if (BLOCKSIZE >= 16)
    {
        sdata[tid] += sdata[tid + 8];
    }
    if (BLOCKSIZE >= 8)
    {
        sdata[tid] += sdata[tid + 4];
    }
    if (BLOCKSIZE >= 4)
    {
        sdata[tid] += sdata[tid + 2];
    }
    if (BLOCKSIZE >= 2)
    {
        sdata[tid] += sdata[tid + 1];
    }
}

template <int BLOCKSIZE>
__global__ void reduce_unroll_all(int *arr, int *out, int len)
{
    __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;        // 线程 id (block 内)
    int bid = blockIdx.x;         // block id (grid 内)
    int bdim = blockDim.x;        // block 大小
    int i = bid * bdim * 2 + tid; // 全局 id

    // 将数据拷贝到共享内存
    if (i < len)
    {
        sdata[tid] = arr[i] + arr[i + bdim];
    }

    __syncthreads(); // 等待所有线程完成

    if (BLOCKSIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCKSIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCKSIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce<BLOCKSIZE>(sdata, tid);
    }

    // 每个 block 的第一个线程将结果写入到 out 中
    if (tid == 0)
    {
        out[bid] = sdata[0];
    }
}

int main()
{
    int *arr = new int[len];
    int *out = new int[len];
    int *d_arr, *d_out;

    // 初始化数组
    for (int i = 0; i < len; i++)
    {
        arr[i] = 1;
    }

    // 分配内存
    cudaMalloc((void **)&d_arr, sizeof(int) * len);
    cudaMalloc((void **)&d_out, sizeof(int) * len);

    // 拷贝数据到显存
    cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);

    // 计算 block 和 grid 的大小
    const int blocksize = 256;
    const int gridsize = (len + blocksize - 1) / (blocksize * 2);

    // 调用 kernel 函数
    reduce_unroll_all<blocksize><<<gridsize, blocksize>>>(d_arr, d_out, len);

    // 拷贝数据到内存
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

    // 计算结果
    long long sum = 0;
    for (int i = 0; i < gridsize; i++)
    {
        sum += out[i];
    }
    printf("sum = %d\n", sum);

    // 核对结果
    long long sum2 = 0;
    for (int i = 0; i < len; i++)
    {
        sum2 += arr[i];
    }

    if (sum == sum2)
    {
        printf("success\n");
    }
    else
    {
        printf("failed, the result is %d\n", sum2);
    }

    // 释放内存
    cudaFree(d_arr);
    cudaFree(d_out);
    delete[] arr;
    delete[] out;
    return 0;
}
