# nvprof 性能分析

## 1. 简介

上一节我们手写了第一个 CUDA Kernel，算是牛刀小试。但我们其实并不知道我们写的 Kernel 执行的效率和性能如何，这就涉及了性能调优。

既然是「调优」，我们首先需要知道 CUDA Kernel 目前的耗时分布以及执行的性能瓶颈在哪里，因此就需要借助性能分析工具帮助我们获取到详细的执行信息，它就是 Nvidia 提供的一个命令行分析工具：`nvprof` 。

## 2. 用法

上一节我们使用 `nvcc ./vector_add.cu -o add` 命令生成了可执行文件，只需要在执行命令前面加上 `nvprof`，即执行 `nvprof ./add`，将会在终端中打印如下信息：

```bash
==33356== Profiling application: ./add
==33356== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   92.23%  570.25ms         1  570.25ms  570.25ms  570.25ms  add_kernel(float*, float*, float*, int)
                    4.79%  29.586ms         1  29.586ms  29.586ms  29.586ms  [CUDA memcpy DtoH]
                    2.99%  18.459ms         2  9.2297ms  9.2245ms  9.2349ms  [CUDA memcpy HtoD]
      API calls:   56.06%  619.64ms         3  206.55ms  9.4402ms  600.73ms  cudaMemcpy
                   43.58%  481.72ms         3  160.57ms  359.50us  481.00ms  cudaMalloc
                    0.16%  1.7937ms       101  17.759us     239ns  933.68us  cuDeviceGetAttribute
                    0.09%  1.0061ms         3  335.36us  278.68us  444.81us  cudaFree
                    0.09%  956.79us         1  956.79us  956.79us  956.79us  cuDeviceTotalMem
                    0.01%  132.25us         1  132.25us  132.25us  132.25us  cuDeviceGetName
                    0.00%  50.300us         1  50.300us  50.300us  50.300us  cudaLaunchKernel
                    0.00%  14.994us         1  14.994us  14.994us  14.994us  cudaDeviceSynchronize
                    0.00%  10.974us         1  10.974us  10.974us  10.974us  cuDeviceGetPCIBusId
                    0.00%  3.0460us         3  1.0150us     421ns  2.1590us  cuDeviceGetCount
                    0.00%  1.7330us         2     866ns     328ns  1.4050us  cuDeviceGet
                    0.00%     543ns         1     543ns     543ns     543ns  cuDeviceGetUuid
```

`nvprof` 还有很多参数可以指定，这个我们稍后再学习。我们学习下如何看懂它给出的执行信息。

## 3. 分析

我们逐行分析上面的日志输出，其中第一行给出的是被分析的程序名 `./add`，即是我们前面 nvcc 编译生成的可执行文件：
```bash
==8936== Profiling application: ./add
```

第二部分是执行可执行文件时，GPU 各个主要「行为」的耗时占比、具体时间、调用次数、平均/最小/最大耗时，接口行为名称：

```bash
           Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   92.23%  570.25ms        1  570.25ms  570.25ms  570.25ms  add_kernel(float*, float*, float*, int)
                  4.79%  29.586ms         1  29.586ms  29.586ms  29.586ms  [CUDA memcpy DtoH]
                  2.99%  18.459ms         2  9.2297ms  9.2245ms  9.2349ms  [CUDA memcpy HtoD]
```

可以看出我们写的 CUDA 程序在 GPU 上主要包括 3 个关键活动：

+ `add_kernel`：即执行 kernel 的时间，占比 92%，耗时 570.25 ms
+ HtoD 的内存拷贝：即输入 `x` &rarr; `cuda_x`，`y` &rarr; `cuda_y` 的 2 次拷贝，占比 2.99%，耗时 18.459 ms
+ DtoH 的内存拷贝：即输出 `cuda_out` &rarr; `out` 的 1 次拷贝，占比 4.79%，耗时 29.586ms 


第三个部分是 CUDA API 的具体调用开销，这个是从 API 层面来解读各个阶段的耗时：

```bash
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
      API calls:   56.06%  619.64ms         3  206.55ms  9.4402ms  600.73ms  cudaMemcpy
                   43.58%  481.72ms         3  160.57ms  359.50us  481.00ms  cudaMalloc
                    0.16%  1.7937ms       101  17.759us     239ns  933.68us  cuDeviceGetAttribute
                    0.09%  1.0061ms         3  335.36us  278.68us  444.81us  cudaFree
                    0.09%  956.79us         1  956.79us  956.79us  956.79us  cuDeviceTotalMem
                    0.01%  132.25us         1  132.25us  132.25us  132.25us  cuDeviceGetName
                    0.00%  50.300us         1  50.300us  50.300us  50.300us  cudaLaunchKernel
                    0.00%  14.994us         1  14.994us  14.994us  14.994us  cudaDeviceSynchronize
                    0.00%  10.974us         1  10.974us  10.974us  10.974us  cuDeviceGetPCIBusId
                    0.00%  3.0460us         3  1.0150us     421ns  2.1590us  cuDeviceGetCount
                    0.00%  1.7330us         2     866ns     328ns  1.4050us  cuDeviceGet
                    0.00%     543ns         1     543ns     543ns     543ns  cuDeviceGetUuid
```

其中最耗时的就是 3 次 `cudaMemcpy` 和`cudasMalloc` 的调用，99% 的时间都在干这两个事情，可以看出显存分配是一个比较「重」的操作，任何时候我们都应该尽量避免频繁的显存分配操作。在深度学习框架中，常会借助「内存池」技术一次申请较大的显存块，然后自己管理切分、分配和回收，这样就可以减少向系统 `cudaMalloc` 的次数，感兴趣的同学可以参考[Paddle 源码之内存管理技术](https://www.cnblogs.com/CocoML/p/14105729.html)。

剩下的 API 调用的开销基本差别不是特别大，大多数都是在 us 级别，我们一一介绍各个 API 的作用：

+ `cuDeviceGetAttribute` : 用于获取 CUDA 设备信息，比如 `cuDeviceGetAttribute(&val, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device)`，参考[支持的 Attribute 列表](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)
+ `cuDeviceTotalMem` : 返回指定显卡的容量，单位 bytes
+ `cudaFree` : 释放申请的显存数据区，归还给系统
+ `cuDeviceGetName` : 返回指定显卡的唯一标识字符串
+ `cudaMemcpy` : 用户设备之间的数据拷贝，可以是 HtoD、DtoH、DtoD
+ `cudaLaunchKernel` : 用于拉起一个函数到 GPU 上去执行，完整的函数签名是：`​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )`。这里需要对此函数的耗时有一个感知，即在几十 us 范围内
+ `cudaDeviceSynchronize` : 用于触发设备的强制同步，即会阻塞住直到设备上所有的计算都做完

## 4. 更多用法

在终端下使用 `nvprof --help` 可以看到更多的参数用法：

```bash
-o,  --export-profile <filename> : 可以选择导出到指定文件，后续可以被其他分析工具可视化

--analysis-metrics : 搭配 --export-profile 使用，用于收集详细 profiling 信息

--trace <gpu|api> : Specify the option (or options seperated by commas) to be traced.

--cpu-profiling <on|off> : Turn on CPU profiling. Note: CPU profiling is not supported in multi-process mode.
```


## 附参考资料

+ [Nvidia 官方 nvprof 使用文档](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof)