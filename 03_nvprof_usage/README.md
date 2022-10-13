# nvprof 性能分析

## 1. 简介

上一节我们手写了第一个 CUDA Kernel，算是牛刀小试。但我们其实并不知道我们写的 Kernel 执行的效率和性能如何，这就涉及了性能调优。

既然是「调优」，我们首先需要知道 CUDA Kernel 目前的耗时分布以及执行的性能瓶颈在哪里，因此就需要借助性能分析工具帮助我们获取到详细的执行信息，它就是 Nvidia 提供的一个命令行分析工具：`nvprof` 。

## 2. 用法

上一节我们使用 `nvcc ./vector_add.cu -o add` 命令生成了可执行文件，只需要在执行命令前面加上 `nvprof`，即执行 `nvprof ./add`，将会在终端中打印如下信息：

```bash
==8936== Profiling application: ./add
==8936== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   89.00%  55.936us         1  55.936us  55.936us  55.936us  add_kernel(float*, float*, float*, int)
                    7.08%  4.4480us         2  2.2240us  2.1760us  2.2720us  [CUDA memcpy HtoD]
                    3.92%  2.4640us         1  2.4640us  2.4640us  2.4640us  [CUDA memcpy DtoH]
      API calls:   99.35%  379.21ms         3  126.40ms  6.4140us  379.20ms  cudaMalloc
                    0.29%  1.1044ms       101  10.934us     235ns  428.75us  cuDeviceGetAttribute
                    0.18%  701.73us         1  701.73us  701.73us  701.73us  cuDeviceTotalMem
                    0.09%  327.20us         3  109.07us  5.5250us  309.13us  cudaFree
                    0.03%  131.37us         1  131.37us  131.37us  131.37us  cuDeviceGetName
                    0.03%  115.67us         3  38.556us  8.5120us  73.420us  cudaMemcpy
                    0.01%  40.429us         1  40.429us  40.429us  40.429us  cudaLaunchKernel
                    0.01%  23.405us         1  23.405us  23.405us  23.405us  cudaDeviceSynchronize
                    0.00%  9.1040us         1  9.1040us  9.1040us  9.1040us  cuDeviceGetPCIBusId
                    0.00%  5.8920us         3  1.9640us     392ns  3.0970us  cuDeviceGetCount
                    0.00%  2.6680us         2  1.3340us     388ns  2.2800us  cuDeviceGet
                    0.00%     477ns         1     477ns     477ns     477ns  cuDeviceGetUuid
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
 GPU activities:   89.00%  55.936us         1  55.936us  55.936us  55.936us  add_kernel(float*, float*, float*, int)
                    7.08%  4.4480us         2  2.2240us  2.1760us  2.2720us  [CUDA memcpy HtoD]
                    3.92%  2.4640us         1  2.4640us  2.4640us  2.4640us  [CUDA memcpy DtoH]
```

可以看出我们写的 CUDA 程序在 GPU 上主要包括 3 个关键活动：

+ `add_kernel`：即执行 kernel 的时间，占比89%，耗时 55.9 us
+ HtoD 的内存拷贝：即输入 `x` &rarr; `cuda_x`，`y` &rarr; `cuda_y` 的 2 次拷贝，占比 7%，耗时 4.4 us
+ DtoH 的内存拷贝：即输出 `cuda_out` &rarr; `out` 的 1次拷贝，占比 3.9%，耗时 2.46 us 


第三个部分是 CUDA API 的具体调用开销，这个是从 API 层面来解读各个阶段的耗时：

```bash
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
      API calls:   99.35%  379.21ms         3  126.40ms  6.4140us  379.20ms  cudaMalloc
                    0.29%  1.1044ms       101  10.934us     235ns  428.75us  cuDeviceGetAttribute
                    0.18%  701.73us         1  701.73us  701.73us  701.73us  cuDeviceTotalMem
                    0.09%  327.20us         3  109.07us  5.5250us  309.13us  cudaFree
                    0.03%  131.37us         1  131.37us  131.37us  131.37us  cuDeviceGetName
                    0.03%  115.67us         3  38.556us  8.5120us  73.420us  cudaMemcpy
                    0.01%  40.429us         1  40.429us  40.429us  40.429us  cudaLaunchKernel
                    0.01%  23.405us         1  23.405us  23.405us  23.405us  cudaDeviceSynchronize
                    0.00%  9.1040us         1  9.1040us  9.1040us  9.1040us  cuDeviceGetPCIBusId
                    0.00%  5.8920us         3  1.9640us     392ns  3.0970us  cuDeviceGetCount
                    0.00%  2.6680us         2  1.3340us     388ns  2.2800us  cuDeviceGet
                    0.00%     477ns         1     477ns     477ns     477ns  cuDeviceGetUuid
```

其中最耗时的就是 3 次 `cudasMalloc` 的调用，99.35% 的时间都在干这个事情，可以看出显存分配是一个比较「重」的操作，任何时候我们都应该尽量避免频繁的显存分配操作。在深度学习框架中，常会借助「内存池」技术一次申请较大的显存块，然后自己管理切分、分配和回收，这样就可以减少向系统 `cudaMalloc` 的次数，感兴趣的同学可以参考[Paddle源码之内存管理技术](https://www.cnblogs.com/CocoML/p/14105729.html)。

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