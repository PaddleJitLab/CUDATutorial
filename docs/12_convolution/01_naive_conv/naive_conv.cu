#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

/*
 * @param n: batch size
 * @param c: 通道数
 * @param h: 输入数据高
 * @param w: 输入数据宽
 * @param k: 卷积核数量
 * @param r: 卷积核高
 * @param s: 卷积核宽
 * @param out_h: 输出数据高
 * @param out_w: 输出数据宽
 * @param u: 卷积在高方向上的步长
 * @param v: 卷积在宽方向上的步长
 * @param p: 卷积在高方向上的补边
 * @param q: 卷积在宽方向上的补边
 * @param in: 输入数据
 * @param weight: 卷积核
 * @param out: 输出数据
 */
__global__ void
naive_conv2d_kernel(int n, int c, int h, int w,
                    int k, int r, int s,
                    int out_h, int out_w,
                    int u, int v, int p, int q,
                    float *in, float *weight, float *out)
{
    // 获取线程在三维网格中的位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    // 如果线程超出工作范围则退出
    if (x >= out_h * out_w || y >= k || z >= n)
    {
        return;
    }

    // 当前线程处理的数据点在out_h、out_w上的坐标
    int pos_out_h = x / out_w;
    int pos_out_w = x % out_w;

    // 计算输入数据的坐标
    int pos_ori_h = pos_out_h * u - p;
    int pos_ori_w = pos_out_w * v - q;

    float sum = 0.0;

    int in_offset = z * c * h * w + pos_ori_h * w + pos_ori_w;
    int weight_offset = y * c * r * s;
    int in_channel_offset = h * w;
    int weight_channel_offset = r * s;

    // 执行卷积操作
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < s; j++)
        {
            int pos_real_h = pos_ori_h + i;
            int pos_real_w = pos_ori_w + j;

            // 只处理有效的数据点
            if (pos_real_h >= 0 && pos_real_w >= 0 && pos_real_w < w && pos_real_h < h)
            {
                int in_offset_tmp = in_offset;
                int wei_offset_tmp = weight_offset;
                for (int channel = 0; channel < c; channel++)
                {
                    // 计算卷积和
                    sum += in[in_offset_tmp + i * w + j] * weight[wei_offset_tmp + i * s + j];
                    in_offset_tmp += in_channel_offset;
                    wei_offset_tmp += weight_channel_offset;
                }
            }
        }
    }

    // 计算输出偏移
    int out_offset = z * k * out_h * out_w + y * out_h * out_w + x;
    out[out_offset] = sum;
}

// CPU 端的卷积计算
void conv2d_cpu(float *in, float *pwei, float *out, int n, int c, int h, int w, int k, int r, int s, int u, int v, int p, int q)
{
    int out_h = (h + 2 * p - r) / u + 1;
    int out_w = (w + 2 * q - s) / v + 1;

    for (int n_num = 0; n_num < n; n_num++)
    {
        for (int k_num = 0; k_num < k; k_num++)
        {
            for (int i = 0; i < out_h; i++)
            {
                for (int j = 0; j < out_w; j++)
                {
                    double sum = 0.0;
                    int pos_h = i * u - p;
                    int pos_w = j * v - q;

                    for (int c_num = 0; c_num < c; c_num++)
                    {
                        for (int kh_num = 0; kh_num < r; kh_num++)
                        {
                            for (int kwNum = 0; kwNum < s; kwNum++)
                            {
                                int pos_ori_h = pos_h + kh_num;
                                int pos_ori_w = pos_w + kwNum;
                                if (pos_ori_w >= 0 && pos_ori_h >= 0 && pos_ori_w < w && pos_ori_h < h)
                                {
                                    sum += (double)(in[n_num * c * h * w + c_num * (w * h) + pos_ori_h * w + pos_ori_w] * pwei[k_num * r * s * c + c_num * r * s + kh_num * s + kwNum]);
                                }
                            }
                        }
                    }

                    out[n_num * k * out_h * out_w + k_num * out_h * out_w + i * out_w + j] = (float)sum;
                }
            }
        }
    }
}

int main()
{
    // 定义输入数据和卷积核的尺寸
    const int n = 2;                           // batch size
    const int c = 2;                           // 通道数
    const int h = 10;                          // 数据高
    const int w = 10;                          // 数据宽
    const int k = 5;                           // 卷积核数量
    const int r = 3;                           // 卷积核高
    const int s = 3;                           // 卷积核宽
    const int out_h = (h - r + 2 * 0) / 1 + 1; // 输出高
    const int out_w = (w - s + 2 * 0) / 1 + 1; // 输出宽
    const int u = 1;                           // 卷积在高方向上的步长
    const int v = 1;                           // 卷积在宽方向上的步长
    const int p = 0;                           // 卷积在高方向上的补边
    const int q = 0;                           // 卷积在宽方向上的补边

    // 分配内存并随机生成输入数据和卷积核
    float *in, *weight, *out;
    float *in_device, *weight_device, *out_device;

    in = (float *)malloc(n * c * h * w * sizeof(float));
    weight = (float *)malloc(k * c * r * s * sizeof(float));
    out = (float *)malloc(n * k * out_h * out_w * sizeof(float));

    cudaMalloc((void **)&in_device, n * c * h * w * sizeof(float));
    cudaMalloc((void **)&weight_device, k * c * r * s * sizeof(float));
    cudaMalloc((void **)&out_device, n * k * out_h * out_w * sizeof(float));

    // 随机生成输入数据和卷积核
    for (int i = 0; i < n * c * h * w; ++i)
    {
        in[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < k * c * r * s; ++i)
    {
        weight[i] = (float)rand() / RAND_MAX;
    }

    // 将输入数据和卷积核拷贝到 GPU
    cudaMemcpy(in_device, in, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_device, out, n * k * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块的大小
    const int blockDim_x = 16;
    const int blockDim_y = 16;

    // 计算线程块和网格的数量
    const int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
    const int gridDim_y = (k + blockDim_y - 1) / blockDim_y;

    // 定义线程块和网

    dim3 blockDim(blockDim_x, blockDim_y);
    dim3 gridDim(gridDim_x, gridDim_y, n);

    // 调用 kernel 函数
    naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in_device, weight_device, out_device);
    // 同步
    cudaDeviceSynchronize();

    // 将 GPU 计算的结果拷贝到 CPU
    cudaMemcpy(out, out_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 端进行卷积计算
    float *out_cpu = (float *)malloc(n * k * out_h * out_w * sizeof(float));
    conv2d_cpu(in, weight, out_cpu, n, c, h, w, k, r, s, u, v, p, q);

    // 比较 GPU 和 CPU 计算结果是否一致
    bool pass = true;
    for (int i = 0; i < n * k * out_h * out_w; ++i)
    {
        if (abs(out[i] - out_cpu[i]) > 1e-5)
        {
            pass = false;
            std::cout << "Verification failed at " << i << "!" << std::endl;
            std::cout << "GPU: " << out_cpu[i] << " CPU: " << out[i] << std::endl;
            break;
        }
    }

    if (pass)
    {
        std::cout << "Verification passed!" << std::endl;

        int iter = 100;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int i = 0; i < iter; i++)
        {
            naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in_device, weight_device, out_device);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "GPU time: " << 1000 * elapsedTime / iter << "us" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 释放内存
    cudaFree(in_device);
    cudaFree(weight_device);
    cudaFree(out_device);
    free(in);
    free(weight);
    free(out);

    return 0;
}