# 展开 Warp

现在我们已经使用了 3 种方法对 Reduce Kernel 进行优化 （交错寻址、去除 Bank Conflilt、减少空闲线程）。
当下实现的 Kernel 距离理论带宽还有一定距离，我们可以继续优化。Reduce 并不是一个算术密集型的 Kernel。
对于这种 Kernel，一个可能的瓶颈就是地址算术指令和循环的开销。

:::note

什么是算术密集型？

算术密集型任务强调的是涉及大量的算术运算，其中包括加法、减法、乘法、除法等基本的数学运算。这类任务通常不涉及复杂的控制流程或数据访问模式，而是侧重于数值计算。图像处理、信号处理和许多科学计算问题都可能属于算术密集型任务。

:::

## 1. 问题分析


在上一个 Kernel 中有如下循环：

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1)
{
    if (tid < s)
    {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

每一次循环都会进行一个 BLOCK 中线程的同步。但是实际上当 `s <= 32` 的时候，由于 `tid <= s` 所以我们只用到了一个 Warp 的线程。由于 cuda 是单指令多线程的设计，所以同一个 Warp 中的线程都是并行执行的。所以最后一个 Warp 在同一个 simd 单元上的这些线程本来就是同步的，所以这个 `__syncthreads()` 同步就是没有必要的了。


## 2. 优化方案

### 2.1. 展开最后一个 Warp

根据前面的分析，我们可以对最后一个 Warp 进行展开，这样就可以减少同步的次数。

```cpp
__device__ void warp_reduce(volatile int *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

注意这里的 `sdata` 是 `volatile` 的，这样可以防止编译器对这些变量进行优化。

:::tip

被 `volatile` 修饰的变量，每次访问都会从内存中读取，而不是从寄存器中读取。这样可以防止编译器对这些变量进行优化。
如果不加 `volatile` 修饰符，编译器会认为这些变量的值不会变化，所以会将这些变量的值缓存在寄存器中。
这样可能导致读到的值不是最新的值。

:::

下面我们就可以对上面的循环进行修改了：

```cpp
for (int s = blockDim.x / 2; s > 32; s >>= 1)
{
    if (tid < s)
    {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

if (tid < 32)
{
    warp_reduce(sdata, tid);
}
```

编译运行命令如下：

```bash
nvcc -o reduce_unroll_last_warp reduce_unroll_last_warp.cu
```

对上面的 Kernel 进行性能分析结果如下：


| 优化手段           | 运行时间(us) | 带宽       | 加速比 |
| ------------------ | ------------ | ---------- | ------ |
| Baseline           | 3118.4       | 42.503GB/s | ~      |
| 交错寻址           | 1904.4       | 73.522GB/s | 1.64   |
| 解决 bank conflict | 1475.2       | 97.536GB/s | 2.29   |
| 去除 idle 线程     | 758.38       | 189.78GB/s | 4.11   |
| 展开最后一个 Warp  | 484.01       | 287.25GB/s | 6.44   |

### 2.2. 完全展开

如果你想追求极致的性能优化，我们可以对 for 循环进行完全展开，这样就可以减少循环的开销。
同时我们可以写一个更加通用的 `warp_reduce` 函数以适用于不同的 BLOCKSIZE。

```cpp
template <unsigned int BLOCKSIZE>
__device__ void warp_reduce(volatile int *sdata, int tid)
{
    if (BLOCKSIZE >= 64)
    {
        sdata[tid] += sdata[tid + 32];
    }
    if (BLOCKSIZE >= 32)
    {
        sdata[tid] += sdata[tid + 16];
    }
    if (BLOCKSIZE >= 16)
    {
        sdata[tid] += sdata[tid + 8];
    }
    if (BLOCKSIZE >= 8)
    {
        sdata[tid] += sdata[tid + 4];
    }
    if (BLOCKSIZE >= 4)
    {
        sdata[tid] += sdata[tid + 2];
    }
    if (BLOCKSIZE >= 2)
    {
        sdata[tid] += sdata[tid + 1];
    }
}
```

下面我们就可以对上面的循环进行修改了：

```cpp
if (BLOCKSIZE >= 512)
{
    if (tid < 256)
    {
        sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
}
if (BLOCKSIZE >= 256)
{
    if (tid < 128)
    {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
}
if (BLOCKSIZE >= 128)
{
    if (tid < 64)
    {
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
}

if (tid < 32)
{
    warp_reduce<BLOCKSIZE>(sdata, tid);
}
```

编译运行命令如下：

```bash
nvcc -o reduce_unroll_all reduce_unroll_all.cu
```

对上面的 Kernel 进行性能分析结果如下：

| 优化手段           | 运行时间(us) | 带宽(GB/s) | 加速比 |
| ------------------ | ------------ | ---------- | ------ |
| Baseline           | 3118.4       | 42.503     | ~      |
| 交错寻址           | 1904.4       | 73.522     | 1.64   |
| 解决 bank conflict | 1475.2       | 97.536     | 2.29   |
| 去除 idle 线程     | 758.38       | 189.78     | 4.11   |
| 展开最后一个 Warp  | 484.01       | 287.25     | 6.44   |
| 完全展开           | 477.23       | 291.77     | 6.53   |


## 3. 总结

在这一节中，我们对 Reduce Kernel 进行了展开 Warp 的优化。
以后我们再写 Kernel 的时候，如果发现有循环的话，可以考虑对循环进行展开，这样可以减少循环的开销。
同时我们也可以考虑有没有不必要的同步，这样可以减少同步的次数，从而提高性能。

## Reference

1. https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
2. http://www.giantpandacv.com/project/OneFlow/%E3%80%90BBuf%E7%9A%84CUDA%E7%AC%94%E8%AE%B0%E3%80%91%E4%B8%89%EF%BC%8Creduce%E4%BC%98%E5%8C%96%E5%85%A5%E9%97%A8%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/

