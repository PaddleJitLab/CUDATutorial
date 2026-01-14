#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

template <int BLOCKSIZE>
__global__ void softmax_kernel_cuda(float *input, float *output, int rows, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= rows) return;

    // Shared Memory 用于 reduction
    __shared__ float s_max[BLOCKSIZE];
    __shared__ float s_sum[BLOCKSIZE];

    // === 第一步：求最大值 ===
    int idx = bid * cols + tid;
    float val = (tid < cols) ? input[idx] : -INFINITY;
    s_max[tid] = val;
    __syncthreads();

    for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float row_max = s_max[0];

    // === 第二步：计算 exp(x - max) 并求和 ===
    val = (tid < cols) ? expf(val - row_max) : 0.0f;
    s_sum[tid] = val;
    __syncthreads();

    for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float row_sum = s_sum[0];

    // === 第三步：归一化并写回 ===
    if (tid < cols) {
        output[idx] = val / row_sum;
    }
}

int main() {
    int rows = 1024;
    int cols = 128;

    // 分配 host 内存
    float *h_input = (float*)malloc(rows * cols * sizeof(float));
    float *h_output = (float*)malloc(rows * cols * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }

    // 分配 device 内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, rows * cols * sizeof(float)));

    // 拷贝数据到 device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // 启动 kernel
    const int BLOCKSIZE = 256;
    dim3 grid(rows);
    dim3 block(BLOCKSIZE);

    softmax_kernel_cuda<BLOCKSIZE><<<grid, block>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝结果回 host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证：检查每行和为 1
    bool correct = true;
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += h_output[i * cols + j];
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            printf("Row %d sum = %f (expected 1.0)\n", i, sum);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("✓ CUDA Softmax verification passed!\n");
    } else {
        printf("✗ CUDA Softmax verification failed!\n");
    }

    // 清理
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
