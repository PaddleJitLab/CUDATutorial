# 构建 CUDA 编程环境

## 1. 前言

进行 CUDA 开发时，首先需要一台带有 GPU 显卡的机器（废话~~），笔记本、台式机、服务器都可以。此仓库以 Linux 系统为基础环境，Windows 环境的配置下文会提供一些教程（我没有 windows，穷~~）。


## 2. Linux 环境搭建

### 2.1 查看 GPU 信息

在装有 GPU 显卡的 Linux 系统上，一般自带了 `nvidia-smi` 命令，可以查看显卡驱动版本号、型号等信息，如下是我开发机的输出信息：

+ CUDA 版本： 11.2
+ 驱动版本： 460.32.03
+ GPU 型号：Tesla 架构 V100
+ 显卡容量：16G * 8 卡


```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:3F:00.0 Off |                    0 |
| N/A   34C    P0    57W / 300W |  16128MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:40:00.0 Off |                    0 |
| N/A   33C    P0    53W / 300W |    764MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:41:00.0 Off |                    0 |
| N/A   36C    P0    54W / 300W |   9666MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:42:00.0 Off |                    0 |
| N/A   37C    P0    56W / 300W |   3280MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   4  Tesla V100-SXM2...  On   | 00000000:62:00.0 Off |                    0 |
| N/A   31C    P0    40W / 300W |      3MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   5  Tesla V100-SXM2...  On   | 00000000:63:00.0 Off |                    0 |
| N/A   31C    P0    39W / 300W |      3MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   6  Tesla V100-SXM2...  On   | 00000000:64:00.0 Off |                    0 |
| N/A   34C    P0    41W / 300W |      3MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   7  Tesla V100-SXM2...  On   | 00000000:65:00.0 Off |                    0 |
| N/A   34C    P0    41W / 300W |      3MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

如果你的机器上显卡驱动都没有安装，可以参考 Nvidia 官网根据你显卡的型号，下载和安装对应的驱动：https://www.nvidia.cn/geforce/drivers/

### 2.2 安装 Toolkit

CUDA Toolkit 是开发 CUDA 程序必备的工具。就像我们写 C++ 一样，你得装 GCC 吧，Toolkit 装完在命令行里输入 `nvcc -V` 就会输出版本信息，比如：

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Jan_28_19:32:09_PST_2021
Cuda compilation tools, release 11.2, V11.2.142
Build cuda_11.2.r11.2/compiler.29558016_0
```

如果还不是很清楚 CUDA Toolkit 是什么，可以翻阅 [Nivida 官网的介绍](https://developer.nvidia.com/cuda-toolkit)：
```
The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to deploy your application.
```

安装时，直接点击 [Nivida 官网](https://developer.nvidia.com/cuda-toolkit) 的 `Download Now` 下载安装即可。安装后可以借助 `nvcc -V` 来确认是否安装成功。


### 2.3 运行 Demo 样例

新建一个 `hello_world.cu` 文件（见此目录）:
```cpp
#include <stdio.h>

__global__ void cuda_say_hello(){
    printf("Hello world, CUDA! %d\n", threadIdx.x);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<1,1>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}
```

首先使用如下命令编译 `nvcc hello_world.cu -o hello_world`, 然后执行 `./hello_world`, 会得到如下输出：

```bash
Hello world, CPU
Hello world, CUDA! 0
```

恭喜你，已经完成的初步环境的搭建了。


### 2.3 Windows 环境配置

Windows 环境配置同样需要安装 CUDA Toolkit，下载地址为：https://developer.nvidia.com/cuda-downloads。

安装成功后可尝试 `nvcc -V` 检测下。和 Linux 不同之处在于，安装 Toolkit 之后还需要配置下环境变量。默认系统会已经有 `CUDA_PATH` 和 `CUDA_PATH_V11.0`（11.0应该是版本号），需要自己在额外添加如下环境变量：

```bash
CUDA_BIN_PATH: %CUDA_PATH%\bin
CUDA_LIB_PATH: %CUDA_PATH%\lib\x64
CUDA_SDK_PATH: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.6  #<---- 注意版本号可能不一样
CUDA_SDK_BIN_PATH: %CUDA_SDK_PATH%\bin\win64
CUDA_SDK_LIB_PATH: %CUDA_SDK_PATH%\common\lib\x64
```

此外，还需要在系统变量 PATH 中添加如下变量：

```bash
%CUDA_BIN_PATH%
%CUDA_LIB_PATH%
%CUDA_SDK_BIN_PATH%
%CUDA_SDK_LIB_PATH%
```

最终，可以运行安装目录下 Nvidia 提供的测试 `.exe` 执行文件：`deviceQuery.exe、bandwidthTest.exe`，如果运行没有问题，则表示环境配置成功了.(在安装路径 `extras/demo_suite`目录里)



## 附参考文档

+ [Say Hello to CUDA 文档](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
+ [printf doesn't output anything in CUDA](https://stackoverflow.com/questions/13320321/printf-in-my-cuda-kernel-doesnt-result-produce-any-output)