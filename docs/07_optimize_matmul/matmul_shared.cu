#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.x;
    const uint c_col = blockIdx.y;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions
    A += c_row * BLOCKSIZE * K;
    B += c_col * BLOCKSIZE;
    C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;

    float tmp = 0.0f;
    for (int i = 0; i < K; i += BLOCKSIZE)
    {
        // load the next block of the input matrices into shared memory
        A_shared[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];

        // wait for all threads to finish loading
        __syncthreads();

        // compute the partial sum
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_col];
        }

        // wait for all threads to finish computing
        __syncthreads();

        // advance the pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    C[thread_row * N + thread_col] = tmp;
}

void run_sgemm_shared_memory(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

int main()
{
    int m = 256;
    int n = 256;
    int k = 256;

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

    run_sgemm_shared_memory(d_A, d_B, d_C, m, n, k);

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

    free(A);
    free(B);
    free(C);
    free(C_ref);
    A = nullptr;
    B = nullptr;
    C = nullptr;
    C_ref = nullptr;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    d_A = nullptr;
    d_B = nullptr;
    d_C = nullptr;

    printf("Success!\n");
    return 0;
}