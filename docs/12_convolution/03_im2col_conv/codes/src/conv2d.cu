#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "conv2d.h"
#include "verfiy.h" // 包含用于验证的自定义头文件

#define KernelErrChk()                                                        \
    {                                                                         \
        cudaError_t errSync = cudaGetLastError();                             \
        cudaError_t errAsync = cudaDeviceSynchronize();                       \
        if (errSync != cudaSuccess)                                           \
        {                                                                     \
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
        if (errAsync != cudaSuccess)                                          \
        {                                                                     \
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

template <const int BM,
          const int BN,
          const int BK,
          const int TM,
          const int TN>
__global__ void sgemm_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C, int kernel_num, int y_height, int y_width)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread; // 一个线程负责计算block中TM*TN个元素

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    当前线程负责搬运全局内存中第a_tile_row行，第a_tile_col列元素至共享内存第a_tile_row行，第a_tile_col列
    a_tile_stride表示block中线程可搬运a_tile_stride行至共享内存；

    若BM=64,BK=8,thread_num=512,则a_tile_stride=64,a_tile_stride=BM，表示每个线程搬运一轮即可完成所需元素的搬运;
    若BM=128,BK=8,thread_num=512,则a_tile_stride=64,表示每个线程搬运两轮即可完成所需元素的搬运;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
    float a_frag[TM] = {0.};
    float b_frag[TN] = {0.};

#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride)
        {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++)
        {
#pragma unroll
            for (int j = 0; j < TM; j++)
            {
                a_frag[j] = As[(ty + j) * BK + i];
            }
#pragma unroll
            for (int l = 0; l < TN; l++)
            {
                b_frag[l] = Bs[tx + l + i * BN];
            }
#pragma unroll
            for (int j = 0; j < TM; j++)
            {
#pragma unroll
                for (int l = 0; l < TN; l++)
                    tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++)
    {
        for (int l = 0; l < TN; l++)
        {
            int batch = l / (y_height * y_width);
            int conv_idx = batch * (kernel_num * y_height * y_width) + ((ty + j) * N + tx) / (batch * y_height * y_width) * (y_height * y_width) + l % (y_height * y_width);
            if ((ty + j) * N + tx + l < M * N)
            {
                C[conv_idx] = alpha * tmp[j][l] + beta * C[conv_idx];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_blocktiling_2d_kernel(float *A,
                                float *B,
                                float *C,
                                int M,
                                int N,
                                int K,
                                int batch,
                                int kernel_num,
                                int y_height,
                                int y_width)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint num_threads_block_tile = (BM * BN) / (TM * TN);

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / (BN / TN);
    const uint thread_col = threadIdx.x % (BN / TN);

    // advance pointers to the starting positions
    A += c_row * BM * K;
    B += c_col * BN;
    int global_c_index = c_row * BM * N + c_col * BN;

    // use to avoid out-of-bounds accesses
    int global_m_pos = c_row * BM * K;
    int global_n_pos = c_col * BN;
    const uint m_size = M * K;
    const uint n_size = N * K;

    assert((BM * BN) / (TM * TN) == blockDim.x);

    const uint A_inner_row = threadIdx.x / BK; // warp-level GMEM coalescing
    const uint A_inner_col = threadIdx.x % BK;
    const uint stride_a = num_threads_block_tile / BK;
    const uint B_inner_row = threadIdx.x / BN; // warp-level GMEM coalescing
    const uint B_inner_col = threadIdx.x % BN;
    const uint stride_b = num_threads_block_tile / BN;

    // allocate thread-local cache for results in registerfile
    float thread_results[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    // outer loop over block tiles
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // load the next block of the input matrices into shared memory
        for (uint load_offset = 0; load_offset < BM; load_offset += stride_a)
        {
            A_shared[(A_inner_row + load_offset) * BK + A_inner_col] =
                (global_m_pos + (A_inner_row + load_offset) * K + A_inner_col < m_size) ? A[(A_inner_row + load_offset) * K + A_inner_col] : 0.0f;
        }
        for (uint load_offset = 0; load_offset < BK; load_offset += stride_b)
        {
            B_shared[(B_inner_row + load_offset) * BN + B_inner_col] =
                (global_n_pos + (B_inner_row + load_offset) * N + B_inner_col < n_size) ? B[(B_inner_row + load_offset) * N + B_inner_col] : 0.0f;
        }

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;
        global_m_pos += BK;
        global_n_pos += BK * N;

        // compute the partial sum
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
        {
            // load relevant As & Bs entries into registers
            for (uint i = 0; i < TM; i++)
            {
                reg_m[i] = A_shared[(thread_row * TM + i) * BK + dot_idx];
            }
            for (uint i = 0; i < TN; i++)
            {
                reg_n[i] = B_shared[dot_idx * BN + thread_col * TN + i];
            }

            // perform outer product on register cache, accumulate
            // into threadResults
            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
                {
                    thread_results[res_idx_m * TN + res_idx_n] += reg_m[res_idx_m] * reg_n[res_idx_n];
                }
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    int inner_y_size = y_height * y_width;
    int res_inner_index, g_index, batch_id, channel_id, inner_offset;

    int conv_idx;

    if (global_c_index >= M * N)
    {
        return;
    }

    for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
    {
        for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
        {
            if (c_row * BM + thread_row * TM + res_idx_m < M && c_col * BN + thread_col * TN + res_idx_n < N)
            {
                res_inner_index = (thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n;
                g_index = global_c_index + res_inner_index;
                inner_offset = g_index % inner_y_size;
                batch_id = (g_index % (inner_y_size * batch)) / inner_y_size;
                channel_id = g_index / (inner_y_size * batch);
                conv_idx = batch_id * (kernel_num * y_height * y_width) + channel_id * (y_height * y_width) + inner_offset;
                C[conv_idx] = thread_results[res_idx_m * TN + res_idx_n];
            }
        }
    }
}

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void cuda_gemm(float *A,
               float *B,
               float *C,
               int M,
               int N,
               int K,
               int batch,
               int kernel_num,
               int y_height,
               int y_width)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    if (M >= 128 && N >= 128)
    {
        const uint BM = 128;
        const uint BN = 128;
        dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 block_size((BM * BN) / (TM * TN));
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, M, N, K, batch, kernel_num, y_height, y_width);
    }
    else
    {
        const uint BM = 64;
        const uint BN = 64;
        dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 block_size((BM * BN) / (TM * TN));
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
            <<<grid_size, block_size>>>(A, B, C, M, N, K, batch, kernel_num, y_height, y_width);
    }
}

#define MAX_THREADS 1024
template <typename T>
static int FetchMaxBlokcSize(T cuda_kernel, const int share_memory_size = 0)
{
    int minGridSize{0};
    int blockSize{0};
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuda_kernel, share_memory_size, MAX_THREADS);
    return blockSize;
}

#define MaxBlockSize 512

template <typename T>
__global__ void im2col_kernel(const int n,
                              T *data_x,
                              T *data_y,
                              const int batches,
                              const int inner_size_x,
                              const int inner_size_y,
                              const int x_height,
                              const int x_width,
                              const int kernel_height,
                              const int kernel_width,
                              const int pad_height,
                              const int pad_width,
                              const int stride_height,
                              const int stride_width,
                              const int dilation_height,
                              const int dilation_width,
                              const int y_height,
                              const int y_width,
                              const int inner_size_c)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x)
    {
        int batch = index / inner_size_c, idx = index % inner_size_c;
        int w_out = idx % y_width, id = idx / y_width;
        int h_out = id % y_height, channel_in = id / y_height;
        int channel_out = channel_in * kernel_height * kernel_width;
        int h_in = h_out * stride_height - pad_height;
        int w_in = w_out * stride_width - pad_width;
        T *out = data_y + batch * (y_height * y_width) + (channel_out * y_height * batches + h_out) * y_width + w_out;
        T *in = data_x + batch * inner_size_x + (channel_in * x_height + h_in) * x_width + w_in;
        for (int i = 0; i < kernel_height; ++i)
        {
            for (int j = 0; j < kernel_width; ++j)
            {
                int h = h_in + i * dilation_height;
                int w = w_in + j * dilation_width;
                *out = (h >= 0 && w >= 0 && h < x_height && w < x_width)
                           ? in[i * dilation_height * x_width + j * dilation_width]
                           : static_cast<T>(0);
                out += y_height * y_width * batches;
            }
        }
    }
}

template <typename T>
void cuda_im2col(const int batches,
                 const int x_channel,
                 const int x_height,
                 const int x_width,
                 const int y_out_plane,
                 const int y_height,
                 const int y_width,
                 const int kernel_height,
                 const int kernel_width,
                 const int stride_height,
                 const int stride_width,
                 const int dilation_height,
                 const int dilation_width,
                 const int pad_height,
                 const int pad_width,
                 T *x,
                 T *y)
{
    const int inner_size_y = y_out_plane * y_height * y_width;
    const int inner_size_x = x_channel * x_height * x_width;
    const int inner_size_c = x_channel * y_height * y_width;
    const int num_kernels = batches * inner_size_c;

    const int blockSize = std::max(std::min(MaxBlockSize, num_kernels), static_cast<int>(1));
    const int gridSize = (num_kernels + blockSize - 1) / blockSize;

    im2col_kernel<T><<<gridSize, blockSize>>>(num_kernels,
                                              x,
                                              y,
                                              batches,
                                              inner_size_x,
                                              inner_size_y,
                                              x_height,
                                              x_width,
                                              kernel_height,
                                              kernel_width,
                                              pad_height,
                                              pad_width,
                                              stride_height,
                                              stride_width,
                                              dilation_height,
                                              dilation_width,
                                              y_height,
                                              y_width,
                                              inner_size_c);
}

int main(int argc, char **argv)
{
    // 从命令行参数中获取输入参数
    int n = atoi(argv[1]);  // 批大小
    int c = atoi(argv[2]);  // 输入通道数
    int h = atoi(argv[3]);  // 输入高度
    int w = atoi(argv[4]);  // 输入宽度
    int k = atoi(argv[5]);  // 卷积核数
    int r = atoi(argv[6]);  // 卷积核高度
    int s = atoi(argv[7]);  // 卷积核宽度
    int u = atoi(argv[8]);  // 垂直步幅
    int v = atoi(argv[9]);  // 水平步幅
    int p = atoi(argv[10]); // 垂直填充
    int q = atoi(argv[11]); // 水平填充

    // 计算输出特征图的高度和宽度
    int outh = (h - r + 2 * p) / u + 1;
    int outw = (w - s + 2 * q) / v + 1;

    // 分配并初始化输入、权重、输出和主机端输出数据的内存
    float *pIn = (float *)malloc(n * c * h * w * sizeof(float));
    float *pInCol = (float *)malloc(n * c * r * s * outh * outw * sizeof(float));
    float *pWeight = (float *)malloc(k * c * r * s * sizeof(float));
    float *pOut = (float *)malloc(n * k * outh * outw * sizeof(float));
    float *pOut_host = (float *)malloc(n * k * outh * outw * sizeof(float));

    float *pIn_device, *pInCol_device, *pWeight_device, *pOut_device;
    cudaMalloc(&pIn_device, n * c * h * w * sizeof(float));
    cudaMalloc(&pInCol_device, n * c * r * s * outh * outw * sizeof(float));
    cudaMalloc(&pWeight_device, k * c * r * s * sizeof(float));
    cudaMalloc(&pOut_device, n * k * outh * outw * sizeof(float));

    // 随机初始化输入和权重数据
    for (int i = 0; i < n * c * h * w; i++)
    {
        pIn[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * c * r * s * outh * outw; i++)
    {
        pInCol[i] = 0.0;
    }

    for (int i = 0; i < k * c * r * s; i++)
    {
        pWeight[i] = (rand() % 255) / 255.0;
    }

    for (int i = 0; i < n * k * outh * outw; i++)
    {
        pOut[i] = 0.0;
        pOut_host[i] = 0.0;
    }

    // 将输入、权重和输出数据从主机内存复制到设备内存
    cudaMemcpy(pIn_device, pIn, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pInCol_device, pInCol, n * c * r * s * outh * outw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_device, pWeight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)pOut_device, pOut, n * k * outh * outw * sizeof(float), cudaMemcpyHostToDevice);

    /*******************************warm up and get result************************************/
    // 计算
    int batch_size = n;
    int x_channel = c;
    int x_height = h;
    int x_width = w;
    int y_height = outh;
    int y_width = outw;
    int kernel_numbers = k;
    int kernel_height = r;
    int kernel_width = s;
    int stride_height = u;
    int stride_width = v;
    int dilation_height = 1;
    int dilation_width = 1;
    int pad_height = p;
    int pad_width = q;
    int y_out_plane = x_channel * kernel_height * kernel_width;

    cuda_im2col(batch_size,
                x_channel,
                x_height,
                x_width,
                y_out_plane,
                y_height,
                y_width,
                kernel_height,
                kernel_width,
                stride_height,
                stride_width,
                dilation_height,
                dilation_width,
                pad_height,
                pad_width,
                pIn_device,
                pInCol_device);
    KernelErrChk();

    cudaDeviceSynchronize();

    std::cout << kernel_numbers << " " << batch_size * y_height * y_width << " " << x_channel * kernel_height * kernel_width << std::endl;
    cuda_gemm(pWeight_device,
              pInCol_device,
              pOut_device,
              kernel_numbers,
              batch_size * y_height * y_width,
              x_channel * kernel_height * kernel_width,
              batch_size,
              kernel_numbers,
              y_height, y_width);

    KernelErrChk();

    cudaDeviceSynchronize();

    cudaMemcpy(pOut_host, pOut_device, n * k * outh * outw * sizeof(float), cudaMemcpyDeviceToHost);

    /*******************************cost time test************************************/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float time_elapsed = 0.0;

    int iternum = 100;
    for (int i = 0; i < iternum; i++)
    {
        cuda_im2col(batch_size,
                    x_channel,
                    x_height,
                    x_width,
                    y_out_plane,
                    y_height,
                    y_width,
                    kernel_height,
                    kernel_width,
                    stride_height,
                    stride_width,
                    dilation_height,
                    dilation_width,
                    pad_height,
                    pad_width,
                    pIn_device,
                    pInCol_device);
        KernelErrChk();

        cudaDeviceSynchronize();

        cuda_gemm(pWeight_device,
                  pInCol_device,
                  pOut_device,
                  kernel_numbers,
                  batch_size * y_height * y_width,
                  x_channel * kernel_height * kernel_width,
                  batch_size,
                  kernel_numbers,
                  y_height, y_width);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    printf("time: %f us\n", time_elapsed * 1000 / iternum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("===================start verify===================\n");
    // 调用CPU上的卷积函数以验证GPU计算结果
    conv2dcpu(pIn, pWeight, pOut, n, c, h, w, k, r, s, u, v, p, q);

    int error = 0;
    for (int i = 0; i < n * k * outh * outw; i++)
    {
        if (abs(pOut_host[i] - pOut[i]) > getPrecision(pOut[i]))
        {
            printf("error, position:%d, gpuvalue:%f, cpuvalue:%f\n", i, pOut_host[i], pOut[i]);
            error++;
            break;
        }
    }
    printf("================finish,error:%d=========================\n", error);

    // 释放设备和主机内存
    cudaFree(pIn_device);
    cudaFree(pWeight_device);
    cudaFree(pOut_device);

    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_host);

    return 0;
}