#include <stdio.h>
#include <cuda_runtime.h>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <typename T>
void free_resource(T *p_cpu, T *p_cuda)
{
    if (nullptr != p_cpu)
    {
        delete[] p_cpu;
        p_cpu = nullptr;
    }
    if (nullptr != p_cuda)
    {
        cudaFree(p_cuda);
        p_cuda = nullptr;
    }
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
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_blocktiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // Block index
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // Thread index within the block
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Size of the 2D tile (block tile)
    const uint total_results_block_tile = BM * BN;
    // Number of threads needed for a block tile
    const uint number_threads_block_tile = total_results_block_tile / (TM * TN);

    assert(number_threads_block_tile == blockDim.x);

    // Calculate the shared memory index that this thread is responsible for loading
    const uint inner_row_A = threadIdx.x / BK;
    const uint inner_col_A = threadIdx.x % BK;

    // Calculate the number of rows each thread block loads at a time
    const uint stride_A = number_threads_block_tile / BK;

    const uint inner_row_B = threadIdx.x / BN;
    const uint inner_col_B = threadIdx.x % BN;
    const uint stride_B = number_threads_block_tile / BN;

    // Shared memory for matrix A and B
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BN * BK];

    // Initialize thread results and register arrays
    float thread_results[TM * TN] = {0.0};
    float reg_m[TM] = {0.0};
    float reg_n[TN] = {0.0};

    // Adjust pointers for A, B, and C
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // Outer loop
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // Load matrix A and B into shared memory
        for (uint load_offset = 0; load_offset < BM; load_offset += stride_A)
        {
            smem_A[(inner_row_A + load_offset) * BK + inner_col_A] = A[(inner_row_A + load_offset) * K + inner_col_A];
        }

        for (uint load_offset = 0; load_offset < BK; load_offset += stride_B)
        {
            smem_B[(inner_row_B + load_offset) * BN + inner_col_B] = B[(inner_row_B + load_offset) * N + inner_col_B];
        }

        // Synchronize threads in the block
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;

        // Compute dot product
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            // Load matrix A and B into registers
            for (uint i = 0; i < TM; i++)
            {
                reg_m[i] = smem_A[(thread_row * TM + i) * BK + dot_idx];
            }
            for (uint i = 0; i < TN; i++)
            {
                reg_n[i] = smem_B[dot_idx * BN + thread_col * TN + i];
            }
            // Compute multiplication and accumulate results
            for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m)
            {
                for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n)
                {
                    thread_results[reg_idx_m * TN + reg_idx_n] +=
                        reg_m[reg_idx_m] * reg_n[reg_idx_n];
                }
            }
        }

        // Synchronize threads in the block
        __syncthreads();
    }

    // Write results back to matrix C
    for (uint reg_idx_m = 0; reg_idx_m < TM; ++reg_idx_m)
    {
        for (uint reg_idx_n = 0; reg_idx_n < TN; ++reg_idx_n)
        {
            C[(thread_row * TM + reg_idx_m) * N + thread_col * TN + reg_idx_n] =
                thread_results[reg_idx_m * TN + reg_idx_n];
        }
    }
}

void run_sgemm_blocktiling_2d(float *A, float *B, float *C, int m, int n, int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;

    const uint BM = 64;
    const uint BN = 64;
    dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
    dim3 block_size((BM * BN) / (TM * TN));
    sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN>
        <<<grid_size, block_size>>>(A, B, C, m, n, k);
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

    run_sgemm_blocktiling_2d(d_A, d_B, d_C, m, n, k);

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
        run_sgemm_blocktiling_2d(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    float avg_run_time = elapsed_time * 1000 / 100;
    printf("Average run time: %f us\n", avg_run_time);

    free_resource(A, d_A);
    free_resource(B, d_B);
    free_resource(C, d_C);
    free_resource(C_ref, (float *)nullptr);

    return 0;
}