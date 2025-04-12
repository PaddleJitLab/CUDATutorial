#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// void atten_naive_cpu(float *Q,
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

void randomize_data(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

void fill_data(float *mat, int N, float value)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = value;
    }
}

// Copy from https://github.com/tspeterkim/flash-attention-minimal
__global__ void flash_attn_v1_kernel(const float *Q,
                                     const float *K,
                                     const float *V,
                                     const int N,
                                     const int d,
                                     const int Tc,
                                     const int Tr,
                                     const int Bc,
                                     const int Br,
                                     const float softmax_scale,
                                     float *l,
                                     float *m,
                                     float *O)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    const int KV_TILE_SIZE = Bc * d; // size of Kj, Vj
    const int Q_TILE_SIZE = Br * d;  // size of Qi
    // const int S_TILE_SIZE = Br * Bc; // size of Sij = softmax(Qi * Kj^T * softmax_scale)
    float *Qi = sram;
    float *Kj = &sram[Q_TILE_SIZE];
    float *Vj = &sram[Q_TILE_SIZE + KV_TILE_SIZE];
    float *S = &sram[Q_TILE_SIZE + KV_TILE_SIZE * 2];

    // outer loop
    for (int j = 0; j < Tc; j++)
    {
        // Load Kj, Vj from HBM to SRAM
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = K[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (KV_TILE_SIZE * j) + (tx * d) + x];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)
        {
            if (tx < Br)
            {
                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++)
                {
                    Qi[(tx * d) + x] = Q[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + tx];
                float row_l_prev = l[lm_offset + (Br * i) + tx];

                // S = QK^T, row_m = rowmax(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++)
                {
                    float sum = 0;
                    for (int x = 0; x < d; x++)
                    {
                        sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    }
                    sum *= softmax_scale;
                    S[(Bc * tx) + y] = sum;

                    if (sum > row_m)
                        row_m = sum;
                }

                // P = exp(S - row_m), row_l = rowsum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++)
                {
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                    row_l += S[(Bc * tx) + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m_prev, row_m);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, m to HBM
                for (int x = 0; x < d; x++)
                {
                    float pv = 0; // Pij * Vj
                    for (int y = 0; y < Bc; y++)
                    {
                        pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    }
                    O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (Q_TILE_SIZE * i) + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
        }
        __syncthreads();
    }
}

// Naive CPU implementation of attention
void attn_cpu(float *Q,
              float *K,
              float *V,
              int B,
              int nh,
              int N,
              int D,
              float softmax_scale,
              float *O)
{
    // Iterate over batch size
    for (int b = 0; b < B; ++b)
    {
        // Iterate over number of attention heads
        for (int h = 0; h < nh; ++h)
        {
            // Iterate over query tokens (index i)
            for (int i = 0; i < N; ++i)
            {
                // Allocate memory for attention scores for this query token (shape N)
                float *scores = (float *)malloc(N * sizeof(float));
                if (scores == NULL)
                {
                    fprintf(stderr, "Memory allocation failed\n");
                    return;
                }

                // Calculate attention scores between the current query token and all
                // key tokens (index j)
                for (int j = 0; j < N; ++j)
                {
                    float score = 0.0f;
                    // Calculate dot product over the dimension D (index d)
                    for (int d = 0; d < D; ++d)
                    {
                        score += Q[((b * nh + h) * N + i) * D + d] *
                                 K[((b * nh + h) * N + j) * D + d];
                    }
                    scores[j] = score * softmax_scale; // Use the provided softmax_scale
                }

                // Apply safe softmax
                // Find the maximum score
                float max_score = scores[0];
                for (int j = 1; j < N; ++j)
                {
                    if (scores[j] > max_score)
                    {
                        max_score = scores[j];
                    }
                }

                // Calculate exponentiated values and their sum
                float sum_exp = 0.0f;
                float *weights = (float *)malloc(N * sizeof(float));
                if (weights == NULL)
                {
                    fprintf(stderr, "Memory allocation failed\n");
                    free(scores);
                    return;
                }
                for (int j = 0; j < N; ++j)
                {
                    weights[j] = expf(scores[j] - max_score);
                    sum_exp += weights[j];
                }

                // Normalize to get attention weights
                for (int j = 0; j < N; ++j)
                {
                    weights[j] /= sum_exp;
                }

                // Calculate the weighted sum of value vectors and store in O
                for (int d = 0; d < D; ++d)
                {
                    O[((b * nh + h) * N + i) * D + d] = 0.0f;
                    for (int j = 0; j < N; ++j)
                    {
                        O[((b * nh + h) * N + i) * D + d] +=
                            weights[j] * V[((b * nh + h) * N + j) * D + d];
                    }
                }

                // Free temporary memory
                free(scores);
                free(weights);
            }
        }
    }
}

int main()
{
    const int B = 4;   // batch size
    const int nh = 8;  // head number
    const int N = 128; // sequence length
    const int D = 64;  // embedding dimension

    // split kv seq_len to Tc and Q seq_len to Tr
    const int Bc = 32;
    // const int Br = 32;
    const int Br = 16;
    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);

    const float softmax_scale = 1.0 / sqrt(D);

    // Allocate memory
    float *Q = (float *)malloc(B * nh * N * D * sizeof(float));
    float *K = (float *)malloc(B * nh * N * D * sizeof(float));
    float *V = (float *)malloc(B * nh * N * D * sizeof(float));
    float *O = (float *)malloc(B * nh * N * D * sizeof(float));
    float *O_cpu = (float *)malloc(B * nh * N * D * sizeof(float));
    float *l = (float *)malloc(B * nh * N * sizeof(float));
    float *m = (float *)malloc(B * nh * N * sizeof(float));

    // Initialize data
    randomize_data(Q, B * nh * N * D);
    randomize_data(K, B * nh * N * D);
    randomize_data(V, B * nh * N * D);
    fill_data(O, B * nh * N * D, 0.0f);
    fill_data(l, B * nh * N, 0.0f);
    fill_data(m, B * nh * N, -INFINITY);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc((void **)&d_Q, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_K, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_V, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_O, B * nh * N * D * sizeof(float));
    cudaMalloc((void **)&d_l, B * nh * N * sizeof(float));
    cudaMalloc((void **)&d_m, B * nh * N * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_Q, Q, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, B * nh * N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, B * nh * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, B * nh * N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate SRAM size needed per block
    const int sram_size =
        (3 * Bc * D * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n",
           max_sram_size,
           sram_size);

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block

    // Launch kernel
    flash_attn_v1_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, D, Tc, Tr, Bc, Br, softmax_scale, d_l, d_m, d_O);

    // Copy result to host
    cudaMemcpy(O, d_O, B * nh * N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Run cpu flash attention
    attn_cpu(Q, K, V, B, nh, N, D, softmax_scale, O_cpu);

    // Check results
    float max_diff = 0.0f;
    for (int i = 0; i < B * nh * N * D; i++)
    {
        max_diff = fmaxf(max_diff, fabsf(O[i] - O_cpu[i]));
    }

    if (max_diff < 0.0001)
    {
        printf("Results are correct! \n");
    }
    else
    {
        printf("Results are incorrect! Max diff: %f\n", max_diff);
    }

    // Free memory
    free_resource(Q, 0);
    free_resource(K, 0);
    free_resource(V, 0);
    free_resource(O, 0);
    free_resource(O_cpu, 0);
    free_resource(l, 0);
    free_resource(m, 0);
    free_resource(d_Q);
    free_resource(d_K);
    free_resource(d_V);
    free_resource(d_O);
    free_resource(d_l);
    free_resource(d_m);

    return 0;
}
