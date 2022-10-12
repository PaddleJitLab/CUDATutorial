# 手写第一个 Kernel 

## 1. 需求

给定两个 1D 的张量 x 和 y，计算 `out = x + y` 的和，输出新的 1D 张量。

## 2. CPU 版本实现

从任务上来看，就是简单的 `out[i] = x[i] + y[i]` 外套一个 for 就完事了。

我们先简单粗暴地用 C++ 写一个 CPU 版本的：

```cpp
#include <stdio.h>

void add_kernel(float *x, float *y, float *out, int n){
    for (int i = 0; i < n; ++i){
        out[i] = x[i] + y[i];
    }
}

int main(){
    int N = 1000;
    size_t mem_size = sizeof(float) * N;

    float *x, *y, *out;
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    out = static_cast<float*>(malloc(mem_size));

    for(int i = 0; i < N; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    add_kernel(x, y, out, N);

    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    free(x);
    free(y);
    free(out);
}
```

CPU 版本大家都比较熟悉，这里用了 malloc + free 在 heap 上申请主机 host 内存。

### 3. CUDA 版本实现

接下来让我们一起把它改为可以在 GPU 上跑起来的 CUDA Kernel。

首先在 CUDA 世界里，要对数据进行计算，那先得把数据放到 GPU 上，即显存，这叫「入乡随俗」。这个过程涉及了一个「内存搬运」操作，既然是搬运，就要有源数据（即 CPU 上内存），也要有目标数据（即 GPU 上显存），还要有人负责搬运（即设备间拷贝接口），分别对应于：

+ malloc 接口申请 CPU 内存
+ cudaMalloc 接口申请 GPU 显存
+ cudaMemcpy 接口负责设备间拷贝

如下是一个简单的「搬运」过程代码：

```cpp
float *x, *cuda_x;

// Allocate CPU memory
x = static_cast<float*>(malloc(mem_size));
// Allocate CUDA memory
cudaMalloc((void**)&cuda_x, mem_size);
// Copy data from CPU to GPU
cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);
```

按照以上方法，我们可以将 CPU 版本代码中的 x，y 数据都先搬运到 GPU 上。

之后，我们需要对 `add_kernel` 做下修改，只需在函数定义处加一个 `__global__` 前置修饰符即可：

```cpp
__global__ void add_kernel(float *x, float *y, float *out, int n){
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] + y[i];
    }
}
```

此外，在 Host 端启动一个 CUDA kernel 需要特殊的「形式」，即 `<<<M, T>>>`，其中 M 表示一个 grid 有 M 个 thread blocks，T 表示一个 thread block 有 T 个并行 thread：
```cpp
add_kernel<<<1, 1>>>(cuda_x, cuda_y, cuda_out, N);
```

最后一步，如果我们想查看 GPU 上计算出的结果是不是正确的，想 printf 打印出来，我们可以选择用 `cudaMemcpy` 接口把结果从 GPU 上拷贝回 CPU，指定接口最后一个参数的拷贝方向为 `cudaMemcpyDeviceToHost` 即可。

手写第一个 CUDA kernel 就完成了，千万别忘了最后用 `cudaFree()` 释放掉显存。完整代码如下：

```cpp
#include <stdio.h>

__global__ void add_kernel(float *x, float *y, float *out, int n){
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] + y[i];
    }
}

int main(){
    int N = 1000;
    size_t mem_size = sizeof(float) * N;

    float *x, *y, *out;
    float *cuda_x, *cuda_y, *cuda_out;

    // Allocate host CPU memory for x, y
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));

    // Initialize x = 1, y = 2
    for(int i = 0; i < N; ++i){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Allocate Device CUDA memory for cuda_x and cuda_y, copy them.
    cudaMalloc((void**)&cuda_x, mem_size);
    cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cuda_y, mem_size);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);

    // Allocate cuda_out CUDA memory and launch add_kernel
    cudaMalloc((void**)&cuda_out, mem_size);
    add_kernel<<<1, 1>>>(cuda_x, cuda_y, cuda_out, N);

    // Copy result from GPU into CPU
    out = static_cast<float*>(malloc(mem_size));
    cudaMemcpy(out, cuda_out, mem_size, cudaMemcpyDeviceToHost);
    
    // Sync CUDA stream to wait kernel completation
    cudaDeviceSynchronize();

    // Print result and checkout out = 3.
    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    // Free CUDA Memory
    cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_out);

    // Free Host CPU Memory
    free(x);
    free(y);
    free(out);

    return 0;
}
```

### 4. 编译执行

使用 `nvcc ./vector_add.cu -o add` 命令生成可执行文件，然后在终端输入 `./add` 执行 kernel，输出结果如下：

```bash
out[0] = 3.000
out[1] = 3.000
out[2] = 3.000
out[3] = 3.000
out[4] = 3.000
out[5] = 3.000
out[6] = 3.000
out[7] = 3.000
out[8] = 3.000
out[9] = 3.000
```

## 附参考文档

+ [Exercise: Converting vector addition to CUDA](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/#putting-things-in-actions)