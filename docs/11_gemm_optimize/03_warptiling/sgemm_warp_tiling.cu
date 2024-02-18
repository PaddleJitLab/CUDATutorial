#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

const uint WARPSIZE = 32;

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

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int WMITER,
          const int WNITER,
          const int TM,
          const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    // warp tile 的大小
    // WM 是每个 Warp 处理数据的行数，WN 是每个 Warp 处理数据的列数
    // 数据行数 / 迭代次数 = 每次迭代处理的行数
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    // warp 内的线程索引
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE; // [0, 31]
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // 每行4个字节作为一个内存块，当前线程负责第inner_row_a行的第inner_col_a个内存块的搬运
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4) * 4;

    // 每行4个字节作为一个内存块，当前线程负责第inner_row_b行的第inner_col_b个内存块的搬运
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;

    // 一共 BM 行，搬运 ldg_a_num 轮，每轮搬运 stride_a 行
    const int stride_a = (NUM_THREADS * 4) / BK;
    // 一共 BN 行，搬运 ldg_b_num 轮，每轮搬运 stride_b 行
    const int stride_b = NUM_THREADS / (BN / 4);

    // 分配共享内存
    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BK * BN];

    // 计算当前线程负责计算的矩阵的起始位置
    A += c_row * BM * K;
    B += c_col * BN;
    C += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN;

    float thread_results[WMITER * WNITER * TM * TN] = {0.0};
    float reg_a[WMITER * TM] = {0.0}; // 缓存 smem_a
    float reg_b[WNITER * TN] = {0.0}; // 缓存 smem_b

    // 外层循环遍历矩阵块
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // 从全局内存加载 A 到共享内存
        for (uint offset = 0; offset < BM; offset += stride_a)
        {
            // 使用原子操作将 A 数据存储到共享内存中
            const float4 tmp = FETCH_FLOAT4(A[OFFSET(offset + inner_row_a, inner_col_a, K)]);
            smem_a[OFFSET(inner_col_a, offset + inner_row_a, BM)] = tmp.x;
            smem_a[OFFSET(inner_col_a + 1, offset + inner_row_a, BM)] = tmp.y;
            smem_a[OFFSET(inner_col_a + 2, offset + inner_row_a, BM)] = tmp.z;
            smem_a[OFFSET(inner_col_a + 3, offset + inner_row_a, BM)] = tmp.w;
        }

        // 从全局内存加载 B 到共享内存
        for (uint offset = 0; offset < BK; offset += stride_b)
        {
            // 使用原子操作将 B 数据存储到共享内存中
            FETCH_FLOAT4(smem_b[OFFSET(inner_row_b + offset, inner_col_b, BN)]) =
                FETCH_FLOAT4(B[OFFSET(inner_row_b + offset, inner_col_b, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // 计算每个线程的结果
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx)
            {
                for (int m = 0; m < TM; m += 4)
                {
                    // 从共享内存中读取数据到寄存器缓存 reg_a
                    FETCH_FLOAT4(reg_a[warp_sub_row_idx * TM + m]) = FETCH_FLOAT4(
                        smem_a[OFFSET(dot_idx, warp_row * WM + warp_sub_row_idx * WSUBM + thread_row_in_warp * TM + m, BM)]);
                }
            }
            for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx)
            {
                for (int n = 0; n < TN; n += 4)
                {
                    // 从共享内存中读取数据到寄存器缓存 reg_b
                    FETCH_FLOAT4(reg_b[warp_sub_col_idx * TN + n]) = FETCH_FLOAT4(
                        smem_b[OFFSET(dot_idx, warp_col * WN + warp_sub_col_idx * WSUBN + thread_col_in_warp * TN + n, BN)]);
                }
            }

            // 计算每个线程的部分结果
            for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx)
            {
                for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx)
                {
                    for (int m = 0; m < TM; m++)
                    {
                        for (int n = 0; n < TN; n++)
                        {
                            // 计算矩阵乘法结果并累加到 thread_results 数组中
                            thread_results[(warp_sub_row_idx * TM + m) * (WNITER * TN) + (warp_sub_col_idx * TN) + n] += reg_a[warp_sub_row_idx * TM + m] * reg_b[warp_sub_col_idx * TN + n];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // 将线程的结果写入全局内存
    for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx)
    {
        for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx)
        {
            // 计算 C 的内存索引并将结果写入 C
            float *C_interim = C + (warp_sub_row_idx * WSUBM) * N + warp_sub_col_idx * WSUBN;
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n += 4)
                {
                    FETCH_FLOAT4(C_interim[OFFSET(m + thread_row_in_warp * TM, n + thread_col_in_warp * TN, N)]) =
                        FETCH_FLOAT4(thread_results[(warp_sub_row_idx * TM + m) * (WNITER * TN) + (warp_sub_col_idx * TN) + n]);
                }
            }
        }
    }
}

void run_sgemm_warp_tiling(float *A, float *B, float *C, int m, int n, int k)
{
    const uint NUM_THREADS = 128;
    const uint BN = 64;
    const uint BM = 64;
    const uint BK = 8;
    const uint WN = 32;
    const uint WM = 32;
    const uint WNITER = 1;
    const uint TN = 4;
    const uint TM = 4;

    dim3 blockDim(NUM_THREADS);

    constexpr uint NUM_WARPS = NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((BN % WN == 0) and (BM % WM == 0));
    static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

    // threads in warpsubtile
    static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) ==
                  0);
    constexpr uint WMITER =
        (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    // warpsubtile in warptile
    static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

    static_assert((NUM_THREADS * 4) % BK == 0,
                  "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((NUM_THREADS * 4) % BN == 0,
                  "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(BN % (16 * TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(BM % (16 * TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(m, BM));

    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WMITER, WNITER, TM,
                            TN, NUM_THREADS>
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
    float *d_A, *d_B, *d_C, *d_C_ref;

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

    // Copy data to device
    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_C_ref, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C_ref, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_warp_tiling(d_A, d_B, d_C, m, n, k);

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
        run_sgemm_warp_tiling(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);
    return 0;
}