# 初识多线程并行计算

## 基础概念

这里我们系统性地学习下 CUDA 编程中的 Thread、Block、Grid 的概念。

GPU 上一般包含很多流式处理器 SM，每个 SM 是 CUDA 架构中的基本计算单元，其可分为若干（如2~3）个网格，每个网格内包含若干（如65535）个线程块，每个线程块包含若干（如512）个线程，概要地理解的话：

+ `Thread`: 一个 CUDA Kernel 可以被多个 threads 来执行
+ `Block`: 多个 threads 会组成一个 Block，而同一个 block 中的 threads 可以同步，也可以通过shared memory通信
+ `Grid`: 多个 blocks 可以组成一个 Grid

其中，一个 Grid 可以包含多个 Blocks。Blocks 的分布方式可以是一维的，二维，三维的；Block 包含多个 Threads，Threads 的分布方式也可以是一维，二维，三维的。

## 线程索引

在[尝试第一次优化 Kernel](https://cuda.keter.top/first_refine_kernel/)中的多 Block 优化的 `add_kernel` 函数实现中，我们计算了 `tid` 的唯一线程标识：

```cpp
__global__ void add_kernel(float *x, float *y, float *out, int n){
    // 唯一的线程下标
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n) {
        out[tid] = x[tid] + y[tid];
    }
}
``` 

这里的 Block 和 Grid 里的单元都是按照一维的形式来组织的，所以在计算 `tid` 时，我们只要到的了 `.x` 后缀的内建变量。

对于二维、三维的 Block 和 Grid，每个线程的索引的计算公式如下。

### 一维 Grid
Grid 为 一维，Block 为一维：
```cpp
int threadId = blockIdx.x *blockDim.x + threadIdx.x; 
```

Grid 为 一维，Block 为二维：
```cpp
int threadId = blockIdx.x * blockDim.x * blockDim.y + 
              threadIdx.y * blockDim.x + threadIdx.x;  
```

Grid 为 一维，Block 为三维：
```cpp
int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + 
              threadIdx.z * blockDim.y * blockDim.x +
              threadIdx.y * blockDim.x + threadIdx.x;  
```

### 二维 Grid
Grid 为 二维，Block 为一维：
```cpp
int blockId = blockIdx.y * gridDim.x + blockIdx.x;  
int threadId = blockId * blockDim.x + threadIdx.x;  
```

Grid 为 二维，Block 为二维：
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y)  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

Grid 为 二维，Block 为三维：
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

### 三维 Grid
Grid 为 三维，Block 为一维：
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x  
             + gridDim.x * gridDim.y * blockIdx.z;  

int threadId = blockId * blockDim.x + threadIdx.x;  
```

Grid 为 三维，Block 为二维：
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x  
             + gridDim.x * gridDim.y * blockIdx.z;  

int threadId = blockId * (blockDim.x * blockDim.y)  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  
```

Grid 为 三维，Block 为三维：
```cpp
int blockId = blockIdx.x + blockIdx.y * gridDim.x  
             + gridDim.x * gridDim.y * blockIdx.z;  

int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;     
```