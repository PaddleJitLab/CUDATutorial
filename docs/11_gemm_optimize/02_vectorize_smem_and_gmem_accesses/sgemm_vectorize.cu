/*
 * 代码主要参考：https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE/blob/master/src/kernel/kernel_6.cuh
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void free_resource(float *ptr, int is_cuda = 1)
{
    if (nullptr != ptr)
    {
        if (is_cuda)
        {
            cudaFree(ptr);
        }
        else
        {
            delete[] ptr;
        }
    }
    ptr = nullptr;
}

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
            {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

// Template parameters:
// BM, BN, BK: dimensions of the block
// TM: number of threads per block
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_vectorize_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const int block_row_thread = BM / TM;
    const int block_col_thread = BN / TN;
    // 一个线程负责计算 block 中 TM*TN 个元素
    const int thread_num = block_row_thread * block_col_thread;

    // 当前线程对应 thread tile 的左上角元素在 block 中的位置
    const int thread_col = (threadIdx.x % block_row_thread) * TN;
    const int thread_row = (threadIdx.x / block_row_thread) * TM;

    // 每行4个字节作为一个内存块，当前线程负责第inner_row_a行的第inner_col_a个内存块的搬运
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4) * 4;

    // 每行4个字节作为一个内存块，当前线程负责第inner_row_b行的第inner_col_b个内存块的搬运
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;

    // 每个线程搬运4个浮点数，完成搬运至 smem_a 需要所有线程搬运 ldg_a_num 轮
    const int ldg_a_num = BK * BM / thread_num / 4;
    // 每个线程搬运4个浮点数，完成搬运至 smem_b 需要所有线程搬运 ldg_b_num 轮
    const int ldg_b_num = BK * BN / thread_num / 4;

    // 一共 BM 行，搬运 ldg_a_num 轮，每轮搬运 stride_a 行
    const int stride_a = BM / ldg_a_num;
    // 一共 BN 行，搬运 ldg_b_num 轮，每轮搬运 stride_b 行
    const int stride_b = BK / ldg_b_num;

    // 分配共享内存
    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BK * BN];

    // 计算当前线程负责计算的矩阵的起始位置
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    float thread_results[TM * TN] = {0.0};
    // 转置时，只用大小为 4 的数组就可以
    float ldg_reg_a[4] = {0.};
    float reg_a[TM] = {0.0}; // 缓存 smem_a
    float reg_b[TN] = {0.0}; // 缓存 smem_b

    // outer-most loop over block tiles
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        for (int i = 0; i < BM; i += stride_a)
        {
            FETCH_FLOAT4(ldg_reg_a[0]) = FETCH_FLOAT4(A[OFFSET(i + inner_row_a, inner_col_a, K)]);
            // smem_a 转置存，其中 ldg_reg_a 做中间缓存，目的是读取时可以按FLOAT4读取
            smem_a[OFFSET(inner_col_a, i + inner_row_a, BM)] = ldg_reg_a[0];
            smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = ldg_reg_a[1];
            smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = ldg_reg_a[2];
            smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = ldg_reg_a[3];
        }

        for (int i = 0; i < BK; i += stride_b)
        {
            FETCH_FLOAT4(smem_b[OFFSET(inner_row_b + i, inner_col_b, BN)]) =
                FETCH_FLOAT4(B[OFFSET(inner_row_b + i, inner_col_b, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (int m = 0; m < TM; m += 4)
            {
                FETCH_FLOAT4(reg_a[m]) = FETCH_FLOAT4(smem_a[OFFSET(dot_idx, thread_row + m, BM)]);
            }
            for (int n = 0; n < TN; n += 4)
            {
                FETCH_FLOAT4(reg_b[n]) = FETCH_FLOAT4(smem_b[OFFSET(dot_idx, thread_col + n, BN)]);
            }

            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    thread_results[m * TN + n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n += 4)
        {
            FETCH_FLOAT4(C[OFFSET(thread_row + m, thread_col + n, N)]) = FETCH_FLOAT4(thread_results[m * TN + n]);
        }
    }
}

void run_sgemm_vectorize(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    const uint BM = 64;
    const uint BN = 64;

    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(A, B, C, m, n, k);
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

int main(int argc, char *argv[])
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Allocate memory for matrices
    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    // save reference result
    C_ref = new float[m * n];

    // Initialize matrices
    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    // Allocate device memory
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_vectorize(d_A, d_B, d_C, m, n, k);

    // Copy result to host
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Run reference sgemm
    sgemm_naive_cpu(A, B, C_ref, m, n, k);

    // Verify result
    for (int i = 0; i < m * n; i++)
    {
        if (C[i] != C_ref[i])
        {
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], C[i]);
            return 1;
        }
    }
    printf("Success!\n");

    // Calculate performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        run_sgemm_vectorize(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);

    free_resource(A, 0);
    free_resource(B, 0);
    free_resource(C, 0);
    free_resource(C_ref, 0);

    free_resource(d_A, 1);
    free_resource(d_B, 1);
    free_resource(d_C, 1);

    return 0;
}