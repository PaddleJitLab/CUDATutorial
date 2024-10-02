# 打印线程号相关信息

本章节旨在帮助用户了解 cuda 内部线程块划分的规则，理解线程号的计算逻辑。

## 1. 1 维 block 和 1 维 thread

样例中设置了两个 block,每个 block 中 64 个线程,  blockDim.x = 64,  
blockIdx.x 代表当前线程所在第几个 block;  
threadIdx.x 代表当前现在在当前 block 中是第几个 thread;  
warp_idx 代表当前线程在当前 block 中是第几个 warp（warp 会选择相邻的线程号做组合）;  
calc_idx 代表当前线程计算的是全局的第几个 thread;  
block 的索引 * 每个 block 的 thread 个数 + block 内的 thread 索引 计算出全局索引。  

```c++
   const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

编译命令 

```bash
nvcc my_id.cu -o my_id
```

执行命令

```bash
./my_id
```

运行结果

```bash
cac_thread   0 - block  0 - warp   0 - thread  0
cac_thread   1 - block  0 - warp   0 - thread  1
cac_thread   2 - block  0 - warp   0 - thread  2
cac_thread   3 - block  0 - warp   0 - thread  3
cac_thread   4 - block  0 - warp   0 - thread  4
cac_thread   5 - block  0 - warp   0 - thread  5
cac_thread   6 - block  0 - warp   0 - thread  6
cac_thread   7 - block  0 - warp   0 - thread  7
cac_thread   8 - block  0 - warp   0 - thread  8
cac_thread   9 - block  0 - warp   0 - thread  9
cac_thread  10 - block  0 - warp   0 - thread 10
cac_thread  11 - block  0 - warp   0 - thread 11
cac_thread  12 - block  0 - warp   0 - thread 12
cac_thread  13 - block  0 - warp   0 - thread 13
cac_thread  14 - block  0 - warp   0 - thread 14
cac_thread  15 - block  0 - warp   0 - thread 15
cac_thread  16 - block  0 - warp   0 - thread 16
cac_thread  17 - block  0 - warp   0 - thread 17
cac_thread  18 - block  0 - warp   0 - thread 18
cac_thread  19 - block  0 - warp   0 - thread 19
cac_thread  20 - block  0 - warp   0 - thread 20
cac_thread  21 - block  0 - warp   0 - thread 21
cac_thread  22 - block  0 - warp   0 - thread 22
cac_thread  23 - block  0 - warp   0 - thread 23
cac_thread  24 - block  0 - warp   0 - thread 24
cac_thread  25 - block  0 - warp   0 - thread 25
cac_thread  26 - block  0 - warp   0 - thread 26
cac_thread  27 - block  0 - warp   0 - thread 27
cac_thread  28 - block  0 - warp   0 - thread 28
cac_thread  29 - block  0 - warp   0 - thread 29
cac_thread  30 - block  0 - warp   0 - thread 30
cac_thread  31 - block  0 - warp   0 - thread 31

cac_thread  32 - block  0 - warp   1 - thread 32
cac_thread  33 - block  0 - warp   1 - thread 33
cac_thread  34 - block  0 - warp   1 - thread 34
cac_thread  35 - block  0 - warp   1 - thread 35
cac_thread  36 - block  0 - warp   1 - thread 36
cac_thread  37 - block  0 - warp   1 - thread 37
cac_thread  38 - block  0 - warp   1 - thread 38
cac_thread  39 - block  0 - warp   1 - thread 39
cac_thread  40 - block  0 - warp   1 - thread 40
cac_thread  41 - block  0 - warp   1 - thread 41
cac_thread  42 - block  0 - warp   1 - thread 42
cac_thread  43 - block  0 - warp   1 - thread 43
cac_thread  44 - block  0 - warp   1 - thread 44
cac_thread  45 - block  0 - warp   1 - thread 45
cac_thread  46 - block  0 - warp   1 - thread 46
cac_thread  47 - block  0 - warp   1 - thread 47
cac_thread  48 - block  0 - warp   1 - thread 48
cac_thread  49 - block  0 - warp   1 - thread 49
cac_thread  50 - block  0 - warp   1 - thread 50
cac_thread  51 - block  0 - warp   1 - thread 51
cac_thread  52 - block  0 - warp   1 - thread 52
cac_thread  53 - block  0 - warp   1 - thread 53
cac_thread  54 - block  0 - warp   1 - thread 54
cac_thread  55 - block  0 - warp   1 - thread 55
cac_thread  56 - block  0 - warp   1 - thread 56
cac_thread  57 - block  0 - warp   1 - thread 57
cac_thread  58 - block  0 - warp   1 - thread 58
cac_thread  59 - block  0 - warp   1 - thread 59
cac_thread  60 - block  0 - warp   1 - thread 60
cac_thread  61 - block  0 - warp   1 - thread 61
cac_thread  62 - block  0 - warp   1 - thread 62
cac_thread  63 - block  0 - warp   1 - thread 63

cac_thread  64 - block  1 - warp   0 - thread  0
cac_thread  65 - block  1 - warp   0 - thread  1
cac_thread  66 - block  1 - warp   0 - thread  2
cac_thread  67 - block  1 - warp   0 - thread  3
cac_thread  68 - block  1 - warp   0 - thread  4
cac_thread  69 - block  1 - warp   0 - thread  5
cac_thread  70 - block  1 - warp   0 - thread  6
cac_thread  71 - block  1 - warp   0 - thread  7
cac_thread  72 - block  1 - warp   0 - thread  8
cac_thread  73 - block  1 - warp   0 - thread  9
cac_thread  74 - block  1 - warp   0 - thread 10
cac_thread  75 - block  1 - warp   0 - thread 11
cac_thread  76 - block  1 - warp   0 - thread 12
cac_thread  77 - block  1 - warp   0 - thread 13
cac_thread  78 - block  1 - warp   0 - thread 14
cac_thread  79 - block  1 - warp   0 - thread 15
cac_thread  80 - block  1 - warp   0 - thread 16
cac_thread  81 - block  1 - warp   0 - thread 17
cac_thread  82 - block  1 - warp   0 - thread 18
cac_thread  83 - block  1 - warp   0 - thread 19
cac_thread  84 - block  1 - warp   0 - thread 20
cac_thread  85 - block  1 - warp   0 - thread 21
cac_thread  86 - block  1 - warp   0 - thread 22
cac_thread  87 - block  1 - warp   0 - thread 23
cac_thread  88 - block  1 - warp   0 - thread 24
cac_thread  89 - block  1 - warp   0 - thread 25
cac_thread  90 - block  1 - warp   0 - thread 26
cac_thread  91 - block  1 - warp   0 - thread 27
cac_thread  92 - block  1 - warp   0 - thread 28
cac_thread  93 - block  1 - warp   0 - thread 29
cac_thread  94 - block  1 - warp   0 - thread 30
cac_thread  95 - block  1 - warp   0 - thread 31

cac_thread  96 - block  1 - warp   1 - thread 32
cac_thread  97 - block  1 - warp   1 - thread 33
cac_thread  98 - block  1 - warp   1 - thread 34
cac_thread  99 - block  1 - warp   1 - thread 35
cac_thread 100 - block  1 - warp   1 - thread 36
cac_thread 101 - block  1 - warp   1 - thread 37
cac_thread 102 - block  1 - warp   1 - thread 38
cac_thread 103 - block  1 - warp   1 - thread 39
cac_thread 104 - block  1 - warp   1 - thread 40
cac_thread 105 - block  1 - warp   1 - thread 41
cac_thread 106 - block  1 - warp   1 - thread 42
cac_thread 107 - block  1 - warp   1 - thread 43
cac_thread 108 - block  1 - warp   1 - thread 44
cac_thread 109 - block  1 - warp   1 - thread 45
cac_thread 110 - block  1 - warp   1 - thread 46
cac_thread 111 - block  1 - warp   1 - thread 47
cac_thread 112 - block  1 - warp   1 - thread 48
cac_thread 113 - block  1 - warp   1 - thread 49
cac_thread 114 - block  1 - warp   1 - thread 50
cac_thread 115 - block  1 - warp   1 - thread 51
cac_thread 116 - block  1 - warp   1 - thread 52
cac_thread 117 - block  1 - warp   1 - thread 53
cac_thread 118 - block  1 - warp   1 - thread 54
cac_thread 119 - block  1 - warp   1 - thread 55
cac_thread 120 - block  1 - warp   1 - thread 56
cac_thread 121 - block  1 - warp   1 - thread 57
cac_thread 122 - block  1 - warp   1 - thread 58
cac_thread 123 - block  1 - warp   1 - thread 59
cac_thread 124 - block  1 - warp   1 - thread 60
cac_thread 125 - block  1 - warp   1 - thread 61
cac_thread 126 - block  1 - warp   1 - thread 62
cac_thread 127 - block  1 - warp   1 - thread 63
```
## 2.  2 维 block 和 2 维 thread

一，二列是用户调用 kernel 时设置的 block 个数 num_blocks =（1，4）， x 维是 1， y 维是 4；  
三，四列是用户调用 kernel 时设置的每个 block 中 thread 个数 num_threads= （32, 4)  x 维是 32， y 维是 4；  

总的线程数计算为`（gridDim.x * gridDim.y）* （blockDim.x * blockDim.y）` 共计 512 个线程。  

```c++
    griddim_x[thread_idx] = gridDim.x; // 1
    griddim_y[thread_idx] = gridDim.y; // 4
    blockdim_x[thread_idx] = blockDim.x; // 32
    blockdim_y[thread_idx] = blockDim.y; // 4
```

gradDim.x 描述 block 在 x 维上的个数； gradDim.y 描述 block 在 y 维上的个数；   
blcokDim.x 描述每个 block 的 x 维上 thread 的个数；blockDim.y 描述每个 block 的 y 维上 thread 的个数。  

五列描述当前线程计算的是全局的第几个 thread。  

```c++
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx ;
```

推导过程如下：
六，七列分别描述当前线程 blockIdx.x， blockIdx.y，  
八，九列分别描述当前线程 threadIdx.x, threadIdx.y。  

```c++
    blockidx_x[thread_idx] = blockIdx.x;
    blockidx_y[thread_idx] = blockIdx.y;
    threadid_x[thread_idx] = threadIdx.x;
    threadid_y[thread_idx] = threadIdx.y;
```

blockIdx.x: 在 grid 的 x 维上第几个 block, blockIdx.y: grid 的 y 维上第几个块；  
threadIdx.x: 在 block 的 x 维上第几个 thread，threadIdx.y: 在 block 的 y 维上第几 thread。  

十，十一列计算了当前线程在 grid 的 x 维上第几个 thread（idx）, 在 grid 的 y 维上第几个 thread（idy),  

计算方式为当前线程在 grid 中（x/y）维第几个 block * 每个 block（x/y 维）的线程个数 + 在当前 block 中（x/y)维第几个线程。  
```c++
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    thread_x[thread_idx] = idx;
    thread_y[thread_idx] = idy;
```

由此可推导全局的索引 = 每行的 thread 数 * 行数 + 单行的列偏移  

```c++
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx ;
```

编译命令 

```bash
nvcc my_id_dim2.cu -o my_id_dim2
```
执行命令

```bash
./my_id_dim2
```
运行结果

```bash
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   0 - blockidx_x  0 -  blockidx_y  0- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   1 - blockidx_x  0 -  blockidx_y  0- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   2 - blockidx_x  0 -  blockidx_y  0- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   3 - blockidx_x  0 -  blockidx_y  0- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   4 - blockidx_x  0 -  blockidx_y  0- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   5 - blockidx_x  0 -  blockidx_y  0- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   6 - blockidx_x  0 -  blockidx_y  0- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   7 - blockidx_x  0 -  blockidx_y  0- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   8 - blockidx_x  0 -  blockidx_y  0- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   9 - blockidx_x  0 -  blockidx_y  0- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  10 - blockidx_x  0 -  blockidx_y  0- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  11 - blockidx_x  0 -  blockidx_y  0- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  12 - blockidx_x  0 -  blockidx_y  0- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  13 - blockidx_x  0 -  blockidx_y  0- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  14 - blockidx_x  0 -  blockidx_y  0- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  15 - blockidx_x  0 -  blockidx_y  0- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  16 - blockidx_x  0 -  blockidx_y  0- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  17 - blockidx_x  0 -  blockidx_y  0- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  18 - blockidx_x  0 -  blockidx_y  0- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  19 - blockidx_x  0 -  blockidx_y  0- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  20 - blockidx_x  0 -  blockidx_y  0- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  21 - blockidx_x  0 -  blockidx_y  0- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  22 - blockidx_x  0 -  blockidx_y  0- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  23 - blockidx_x  0 -  blockidx_y  0- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  24 - blockidx_x  0 -  blockidx_y  0- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  25 - blockidx_x  0 -  blockidx_y  0- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  26 - blockidx_x  0 -  blockidx_y  0- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  27 - blockidx_x  0 -  blockidx_y  0- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  28 - blockidx_x  0 -  blockidx_y  0- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  29 - blockidx_x  0 -  blockidx_y  0- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  30 - blockidx_x  0 -  blockidx_y  0- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  31 - blockidx_x  0 -  blockidx_y  0- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y  0 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  32 - blockidx_x  0 -  blockidx_y  0- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  33 - blockidx_x  0 -  blockidx_y  0- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  34 - blockidx_x  0 -  blockidx_y  0- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  35 - blockidx_x  0 -  blockidx_y  0- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  36 - blockidx_x  0 -  blockidx_y  0- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  37 - blockidx_x  0 -  blockidx_y  0- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  38 - blockidx_x  0 -  blockidx_y  0- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  39 - blockidx_x  0 -  blockidx_y  0- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  40 - blockidx_x  0 -  blockidx_y  0- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  41 - blockidx_x  0 -  blockidx_y  0- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  42 - blockidx_x  0 -  blockidx_y  0- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  43 - blockidx_x  0 -  blockidx_y  0- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  44 - blockidx_x  0 -  blockidx_y  0- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  45 - blockidx_x  0 -  blockidx_y  0- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  46 - blockidx_x  0 -  blockidx_y  0- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  47 - blockidx_x  0 -  blockidx_y  0- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  48 - blockidx_x  0 -  blockidx_y  0- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  49 - blockidx_x  0 -  blockidx_y  0- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  50 - blockidx_x  0 -  blockidx_y  0- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  51 - blockidx_x  0 -  blockidx_y  0- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  52 - blockidx_x  0 -  blockidx_y  0- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  53 - blockidx_x  0 -  blockidx_y  0- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  54 - blockidx_x  0 -  blockidx_y  0- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  55 - blockidx_x  0 -  blockidx_y  0- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  56 - blockidx_x  0 -  blockidx_y  0- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  57 - blockidx_x  0 -  blockidx_y  0- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  58 - blockidx_x  0 -  blockidx_y  0- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  59 - blockidx_x  0 -  blockidx_y  0- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  60 - blockidx_x  0 -  blockidx_y  0- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  61 - blockidx_x  0 -  blockidx_y  0- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  62 - blockidx_x  0 -  blockidx_y  0- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  63 - blockidx_x  0 -  blockidx_y  0- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y  1 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  64 - blockidx_x  0 -  blockidx_y  0- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  65 - blockidx_x  0 -  blockidx_y  0- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  66 - blockidx_x  0 -  blockidx_y  0- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  67 - blockidx_x  0 -  blockidx_y  0- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  68 - blockidx_x  0 -  blockidx_y  0- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  69 - blockidx_x  0 -  blockidx_y  0- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  70 - blockidx_x  0 -  blockidx_y  0- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  71 - blockidx_x  0 -  blockidx_y  0- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  72 - blockidx_x  0 -  blockidx_y  0- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  73 - blockidx_x  0 -  blockidx_y  0- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  74 - blockidx_x  0 -  blockidx_y  0- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  75 - blockidx_x  0 -  blockidx_y  0- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  76 - blockidx_x  0 -  blockidx_y  0- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  77 - blockidx_x  0 -  blockidx_y  0- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  78 - blockidx_x  0 -  blockidx_y  0- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  79 - blockidx_x  0 -  blockidx_y  0- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  80 - blockidx_x  0 -  blockidx_y  0- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  81 - blockidx_x  0 -  blockidx_y  0- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  82 - blockidx_x  0 -  blockidx_y  0- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  83 - blockidx_x  0 -  blockidx_y  0- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  84 - blockidx_x  0 -  blockidx_y  0- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  85 - blockidx_x  0 -  blockidx_y  0- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  86 - blockidx_x  0 -  blockidx_y  0- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  87 - blockidx_x  0 -  blockidx_y  0- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  88 - blockidx_x  0 -  blockidx_y  0- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  89 - blockidx_x  0 -  blockidx_y  0- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  90 - blockidx_x  0 -  blockidx_y  0- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  91 - blockidx_x  0 -  blockidx_y  0- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  92 - blockidx_x  0 -  blockidx_y  0- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  93 - blockidx_x  0 -  blockidx_y  0- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  94 - blockidx_x  0 -  blockidx_y  0- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  95 - blockidx_x  0 -  blockidx_y  0- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y  2 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  96 - blockidx_x  0 -  blockidx_y  0- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  97 - blockidx_x  0 -  blockidx_y  0- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  98 - blockidx_x  0 -  blockidx_y  0- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  99 - blockidx_x  0 -  blockidx_y  0- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 100 - blockidx_x  0 -  blockidx_y  0- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 101 - blockidx_x  0 -  blockidx_y  0- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 102 - blockidx_x  0 -  blockidx_y  0- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 103 - blockidx_x  0 -  blockidx_y  0- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 104 - blockidx_x  0 -  blockidx_y  0- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 105 - blockidx_x  0 -  blockidx_y  0- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 106 - blockidx_x  0 -  blockidx_y  0- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 107 - blockidx_x  0 -  blockidx_y  0- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 108 - blockidx_x  0 -  blockidx_y  0- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 109 - blockidx_x  0 -  blockidx_y  0- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 110 - blockidx_x  0 -  blockidx_y  0- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 111 - blockidx_x  0 -  blockidx_y  0- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 112 - blockidx_x  0 -  blockidx_y  0- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 113 - blockidx_x  0 -  blockidx_y  0- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 114 - blockidx_x  0 -  blockidx_y  0- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 115 - blockidx_x  0 -  blockidx_y  0- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 116 - blockidx_x  0 -  blockidx_y  0- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 117 - blockidx_x  0 -  blockidx_y  0- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 118 - blockidx_x  0 -  blockidx_y  0- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 119 - blockidx_x  0 -  blockidx_y  0- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 120 - blockidx_x  0 -  blockidx_y  0- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 121 - blockidx_x  0 -  blockidx_y  0- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 122 - blockidx_x  0 -  blockidx_y  0- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 123 - blockidx_x  0 -  blockidx_y  0- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 124 - blockidx_x  0 -  blockidx_y  0- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 125 - blockidx_x  0 -  blockidx_y  0- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 126 - blockidx_x  0 -  blockidx_y  0- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 127 - blockidx_x  0 -  blockidx_y  0- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y  3 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 128 - blockidx_x  0 -  blockidx_y  1- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 129 - blockidx_x  0 -  blockidx_y  1- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 130 - blockidx_x  0 -  blockidx_y  1- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 131 - blockidx_x  0 -  blockidx_y  1- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 132 - blockidx_x  0 -  blockidx_y  1- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 133 - blockidx_x  0 -  blockidx_y  1- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 134 - blockidx_x  0 -  blockidx_y  1- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 135 - blockidx_x  0 -  blockidx_y  1- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 136 - blockidx_x  0 -  blockidx_y  1- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 137 - blockidx_x  0 -  blockidx_y  1- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 138 - blockidx_x  0 -  blockidx_y  1- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 139 - blockidx_x  0 -  blockidx_y  1- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 140 - blockidx_x  0 -  blockidx_y  1- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 141 - blockidx_x  0 -  blockidx_y  1- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 142 - blockidx_x  0 -  blockidx_y  1- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 143 - blockidx_x  0 -  blockidx_y  1- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 144 - blockidx_x  0 -  blockidx_y  1- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 145 - blockidx_x  0 -  blockidx_y  1- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 146 - blockidx_x  0 -  blockidx_y  1- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 147 - blockidx_x  0 -  blockidx_y  1- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 148 - blockidx_x  0 -  blockidx_y  1- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 149 - blockidx_x  0 -  blockidx_y  1- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 150 - blockidx_x  0 -  blockidx_y  1- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 151 - blockidx_x  0 -  blockidx_y  1- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 152 - blockidx_x  0 -  blockidx_y  1- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 153 - blockidx_x  0 -  blockidx_y  1- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 154 - blockidx_x  0 -  blockidx_y  1- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 155 - blockidx_x  0 -  blockidx_y  1- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 156 - blockidx_x  0 -  blockidx_y  1- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 157 - blockidx_x  0 -  blockidx_y  1- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 158 - blockidx_x  0 -  blockidx_y  1- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 159 - blockidx_x  0 -  blockidx_y  1- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y  4 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 160 - blockidx_x  0 -  blockidx_y  1- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 161 - blockidx_x  0 -  blockidx_y  1- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 162 - blockidx_x  0 -  blockidx_y  1- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 163 - blockidx_x  0 -  blockidx_y  1- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 164 - blockidx_x  0 -  blockidx_y  1- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 165 - blockidx_x  0 -  blockidx_y  1- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 166 - blockidx_x  0 -  blockidx_y  1- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 167 - blockidx_x  0 -  blockidx_y  1- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 168 - blockidx_x  0 -  blockidx_y  1- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 169 - blockidx_x  0 -  blockidx_y  1- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 170 - blockidx_x  0 -  blockidx_y  1- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 171 - blockidx_x  0 -  blockidx_y  1- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 172 - blockidx_x  0 -  blockidx_y  1- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 173 - blockidx_x  0 -  blockidx_y  1- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 174 - blockidx_x  0 -  blockidx_y  1- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 175 - blockidx_x  0 -  blockidx_y  1- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 176 - blockidx_x  0 -  blockidx_y  1- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 177 - blockidx_x  0 -  blockidx_y  1- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 178 - blockidx_x  0 -  blockidx_y  1- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 179 - blockidx_x  0 -  blockidx_y  1- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 180 - blockidx_x  0 -  blockidx_y  1- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 181 - blockidx_x  0 -  blockidx_y  1- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 182 - blockidx_x  0 -  blockidx_y  1- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 183 - blockidx_x  0 -  blockidx_y  1- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 184 - blockidx_x  0 -  blockidx_y  1- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 185 - blockidx_x  0 -  blockidx_y  1- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 186 - blockidx_x  0 -  blockidx_y  1- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 187 - blockidx_x  0 -  blockidx_y  1- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 188 - blockidx_x  0 -  blockidx_y  1- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 189 - blockidx_x  0 -  blockidx_y  1- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 190 - blockidx_x  0 -  blockidx_y  1- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 191 - blockidx_x  0 -  blockidx_y  1- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y  5 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 192 - blockidx_x  0 -  blockidx_y  1- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 193 - blockidx_x  0 -  blockidx_y  1- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 194 - blockidx_x  0 -  blockidx_y  1- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 195 - blockidx_x  0 -  blockidx_y  1- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 196 - blockidx_x  0 -  blockidx_y  1- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 197 - blockidx_x  0 -  blockidx_y  1- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 198 - blockidx_x  0 -  blockidx_y  1- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 199 - blockidx_x  0 -  blockidx_y  1- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 200 - blockidx_x  0 -  blockidx_y  1- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 201 - blockidx_x  0 -  blockidx_y  1- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 202 - blockidx_x  0 -  blockidx_y  1- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 203 - blockidx_x  0 -  blockidx_y  1- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 204 - blockidx_x  0 -  blockidx_y  1- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 205 - blockidx_x  0 -  blockidx_y  1- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 206 - blockidx_x  0 -  blockidx_y  1- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 207 - blockidx_x  0 -  blockidx_y  1- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 208 - blockidx_x  0 -  blockidx_y  1- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 209 - blockidx_x  0 -  blockidx_y  1- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 210 - blockidx_x  0 -  blockidx_y  1- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 211 - blockidx_x  0 -  blockidx_y  1- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 212 - blockidx_x  0 -  blockidx_y  1- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 213 - blockidx_x  0 -  blockidx_y  1- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 214 - blockidx_x  0 -  blockidx_y  1- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 215 - blockidx_x  0 -  blockidx_y  1- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 216 - blockidx_x  0 -  blockidx_y  1- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 217 - blockidx_x  0 -  blockidx_y  1- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 218 - blockidx_x  0 -  blockidx_y  1- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 219 - blockidx_x  0 -  blockidx_y  1- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 220 - blockidx_x  0 -  blockidx_y  1- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 221 - blockidx_x  0 -  blockidx_y  1- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 222 - blockidx_x  0 -  blockidx_y  1- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 223 - blockidx_x  0 -  blockidx_y  1- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y  6 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 224 - blockidx_x  0 -  blockidx_y  1- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 225 - blockidx_x  0 -  blockidx_y  1- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 226 - blockidx_x  0 -  blockidx_y  1- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 227 - blockidx_x  0 -  blockidx_y  1- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 228 - blockidx_x  0 -  blockidx_y  1- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 229 - blockidx_x  0 -  blockidx_y  1- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 230 - blockidx_x  0 -  blockidx_y  1- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 231 - blockidx_x  0 -  blockidx_y  1- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 232 - blockidx_x  0 -  blockidx_y  1- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 233 - blockidx_x  0 -  blockidx_y  1- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 234 - blockidx_x  0 -  blockidx_y  1- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 235 - blockidx_x  0 -  blockidx_y  1- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 236 - blockidx_x  0 -  blockidx_y  1- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 237 - blockidx_x  0 -  blockidx_y  1- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 238 - blockidx_x  0 -  blockidx_y  1- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 239 - blockidx_x  0 -  blockidx_y  1- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 240 - blockidx_x  0 -  blockidx_y  1- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 241 - blockidx_x  0 -  blockidx_y  1- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 242 - blockidx_x  0 -  blockidx_y  1- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 243 - blockidx_x  0 -  blockidx_y  1- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 244 - blockidx_x  0 -  blockidx_y  1- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 245 - blockidx_x  0 -  blockidx_y  1- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 246 - blockidx_x  0 -  blockidx_y  1- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 247 - blockidx_x  0 -  blockidx_y  1- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 248 - blockidx_x  0 -  blockidx_y  1- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 249 - blockidx_x  0 -  blockidx_y  1- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 250 - blockidx_x  0 -  blockidx_y  1- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 251 - blockidx_x  0 -  blockidx_y  1- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 252 - blockidx_x  0 -  blockidx_y  1- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 253 - blockidx_x  0 -  blockidx_y  1- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 254 - blockidx_x  0 -  blockidx_y  1- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 255 - blockidx_x  0 -  blockidx_y  1- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y  7 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 256 - blockidx_x  0 -  blockidx_y  2- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 257 - blockidx_x  0 -  blockidx_y  2- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 258 - blockidx_x  0 -  blockidx_y  2- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 259 - blockidx_x  0 -  blockidx_y  2- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 260 - blockidx_x  0 -  blockidx_y  2- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 261 - blockidx_x  0 -  blockidx_y  2- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 262 - blockidx_x  0 -  blockidx_y  2- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 263 - blockidx_x  0 -  blockidx_y  2- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 264 - blockidx_x  0 -  blockidx_y  2- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 265 - blockidx_x  0 -  blockidx_y  2- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 266 - blockidx_x  0 -  blockidx_y  2- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 267 - blockidx_x  0 -  blockidx_y  2- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 268 - blockidx_x  0 -  blockidx_y  2- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 269 - blockidx_x  0 -  blockidx_y  2- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 270 - blockidx_x  0 -  blockidx_y  2- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 271 - blockidx_x  0 -  blockidx_y  2- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 272 - blockidx_x  0 -  blockidx_y  2- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 273 - blockidx_x  0 -  blockidx_y  2- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 274 - blockidx_x  0 -  blockidx_y  2- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 275 - blockidx_x  0 -  blockidx_y  2- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 276 - blockidx_x  0 -  blockidx_y  2- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 277 - blockidx_x  0 -  blockidx_y  2- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 278 - blockidx_x  0 -  blockidx_y  2- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 279 - blockidx_x  0 -  blockidx_y  2- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 280 - blockidx_x  0 -  blockidx_y  2- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 281 - blockidx_x  0 -  blockidx_y  2- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 282 - blockidx_x  0 -  blockidx_y  2- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 283 - blockidx_x  0 -  blockidx_y  2- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 284 - blockidx_x  0 -  blockidx_y  2- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 285 - blockidx_x  0 -  blockidx_y  2- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 286 - blockidx_x  0 -  blockidx_y  2- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 287 - blockidx_x  0 -  blockidx_y  2- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y  8 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 288 - blockidx_x  0 -  blockidx_y  2- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 289 - blockidx_x  0 -  blockidx_y  2- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 290 - blockidx_x  0 -  blockidx_y  2- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 291 - blockidx_x  0 -  blockidx_y  2- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 292 - blockidx_x  0 -  blockidx_y  2- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 293 - blockidx_x  0 -  blockidx_y  2- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 294 - blockidx_x  0 -  blockidx_y  2- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 295 - blockidx_x  0 -  blockidx_y  2- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 296 - blockidx_x  0 -  blockidx_y  2- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 297 - blockidx_x  0 -  blockidx_y  2- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 298 - blockidx_x  0 -  blockidx_y  2- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 299 - blockidx_x  0 -  blockidx_y  2- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 300 - blockidx_x  0 -  blockidx_y  2- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 301 - blockidx_x  0 -  blockidx_y  2- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 302 - blockidx_x  0 -  blockidx_y  2- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 303 - blockidx_x  0 -  blockidx_y  2- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 304 - blockidx_x  0 -  blockidx_y  2- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 305 - blockidx_x  0 -  blockidx_y  2- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 306 - blockidx_x  0 -  blockidx_y  2- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 307 - blockidx_x  0 -  blockidx_y  2- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 308 - blockidx_x  0 -  blockidx_y  2- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 309 - blockidx_x  0 -  blockidx_y  2- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 310 - blockidx_x  0 -  blockidx_y  2- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 311 - blockidx_x  0 -  blockidx_y  2- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 312 - blockidx_x  0 -  blockidx_y  2- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 313 - blockidx_x  0 -  blockidx_y  2- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 314 - blockidx_x  0 -  blockidx_y  2- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 315 - blockidx_x  0 -  blockidx_y  2- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 316 - blockidx_x  0 -  blockidx_y  2- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 317 - blockidx_x  0 -  blockidx_y  2- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 318 - blockidx_x  0 -  blockidx_y  2- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 319 - blockidx_x  0 -  blockidx_y  2- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y  9 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 320 - blockidx_x  0 -  blockidx_y  2- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 321 - blockidx_x  0 -  blockidx_y  2- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 322 - blockidx_x  0 -  blockidx_y  2- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 323 - blockidx_x  0 -  blockidx_y  2- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 324 - blockidx_x  0 -  blockidx_y  2- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 325 - blockidx_x  0 -  blockidx_y  2- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 326 - blockidx_x  0 -  blockidx_y  2- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 327 - blockidx_x  0 -  blockidx_y  2- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 328 - blockidx_x  0 -  blockidx_y  2- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 329 - blockidx_x  0 -  blockidx_y  2- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 330 - blockidx_x  0 -  blockidx_y  2- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 331 - blockidx_x  0 -  blockidx_y  2- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 332 - blockidx_x  0 -  blockidx_y  2- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 333 - blockidx_x  0 -  blockidx_y  2- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 334 - blockidx_x  0 -  blockidx_y  2- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 335 - blockidx_x  0 -  blockidx_y  2- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 336 - blockidx_x  0 -  blockidx_y  2- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 337 - blockidx_x  0 -  blockidx_y  2- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 338 - blockidx_x  0 -  blockidx_y  2- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 339 - blockidx_x  0 -  blockidx_y  2- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 340 - blockidx_x  0 -  blockidx_y  2- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 341 - blockidx_x  0 -  blockidx_y  2- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 342 - blockidx_x  0 -  blockidx_y  2- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 343 - blockidx_x  0 -  blockidx_y  2- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 344 - blockidx_x  0 -  blockidx_y  2- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 345 - blockidx_x  0 -  blockidx_y  2- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 346 - blockidx_x  0 -  blockidx_y  2- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 347 - blockidx_x  0 -  blockidx_y  2- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 348 - blockidx_x  0 -  blockidx_y  2- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 349 - blockidx_x  0 -  blockidx_y  2- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 350 - blockidx_x  0 -  blockidx_y  2- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 351 - blockidx_x  0 -  blockidx_y  2- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 10 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 352 - blockidx_x  0 -  blockidx_y  2- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 353 - blockidx_x  0 -  blockidx_y  2- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 354 - blockidx_x  0 -  blockidx_y  2- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 355 - blockidx_x  0 -  blockidx_y  2- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 356 - blockidx_x  0 -  blockidx_y  2- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 357 - blockidx_x  0 -  blockidx_y  2- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 358 - blockidx_x  0 -  blockidx_y  2- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 359 - blockidx_x  0 -  blockidx_y  2- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 360 - blockidx_x  0 -  blockidx_y  2- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 361 - blockidx_x  0 -  blockidx_y  2- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 362 - blockidx_x  0 -  blockidx_y  2- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 363 - blockidx_x  0 -  blockidx_y  2- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 364 - blockidx_x  0 -  blockidx_y  2- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 365 - blockidx_x  0 -  blockidx_y  2- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 366 - blockidx_x  0 -  blockidx_y  2- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 367 - blockidx_x  0 -  blockidx_y  2- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 368 - blockidx_x  0 -  blockidx_y  2- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 369 - blockidx_x  0 -  blockidx_y  2- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 370 - blockidx_x  0 -  blockidx_y  2- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 371 - blockidx_x  0 -  blockidx_y  2- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 372 - blockidx_x  0 -  blockidx_y  2- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 373 - blockidx_x  0 -  blockidx_y  2- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 374 - blockidx_x  0 -  blockidx_y  2- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 375 - blockidx_x  0 -  blockidx_y  2- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 376 - blockidx_x  0 -  blockidx_y  2- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 377 - blockidx_x  0 -  blockidx_y  2- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 378 - blockidx_x  0 -  blockidx_y  2- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 379 - blockidx_x  0 -  blockidx_y  2- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 380 - blockidx_x  0 -  blockidx_y  2- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 381 - blockidx_x  0 -  blockidx_y  2- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 382 - blockidx_x  0 -  blockidx_y  2- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 383 - blockidx_x  0 -  blockidx_y  2- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 11 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 384 - blockidx_x  0 -  blockidx_y  3- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 385 - blockidx_x  0 -  blockidx_y  3- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 386 - blockidx_x  0 -  blockidx_y  3- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 387 - blockidx_x  0 -  blockidx_y  3- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 388 - blockidx_x  0 -  blockidx_y  3- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 389 - blockidx_x  0 -  blockidx_y  3- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 390 - blockidx_x  0 -  blockidx_y  3- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 391 - blockidx_x  0 -  blockidx_y  3- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 392 - blockidx_x  0 -  blockidx_y  3- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 393 - blockidx_x  0 -  blockidx_y  3- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 394 - blockidx_x  0 -  blockidx_y  3- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 395 - blockidx_x  0 -  blockidx_y  3- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 396 - blockidx_x  0 -  blockidx_y  3- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 397 - blockidx_x  0 -  blockidx_y  3- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 398 - blockidx_x  0 -  blockidx_y  3- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 399 - blockidx_x  0 -  blockidx_y  3- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 400 - blockidx_x  0 -  blockidx_y  3- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 401 - blockidx_x  0 -  blockidx_y  3- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 402 - blockidx_x  0 -  blockidx_y  3- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 403 - blockidx_x  0 -  blockidx_y  3- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 404 - blockidx_x  0 -  blockidx_y  3- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 405 - blockidx_x  0 -  blockidx_y  3- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 406 - blockidx_x  0 -  blockidx_y  3- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 407 - blockidx_x  0 -  blockidx_y  3- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 408 - blockidx_x  0 -  blockidx_y  3- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 409 - blockidx_x  0 -  blockidx_y  3- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 410 - blockidx_x  0 -  blockidx_y  3- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 411 - blockidx_x  0 -  blockidx_y  3- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 412 - blockidx_x  0 -  blockidx_y  3- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 413 - blockidx_x  0 -  blockidx_y  3- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 414 - blockidx_x  0 -  blockidx_y  3- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 415 - blockidx_x  0 -  blockidx_y  3- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y 12 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 416 - blockidx_x  0 -  blockidx_y  3- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 417 - blockidx_x  0 -  blockidx_y  3- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 418 - blockidx_x  0 -  blockidx_y  3- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 419 - blockidx_x  0 -  blockidx_y  3- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 420 - blockidx_x  0 -  blockidx_y  3- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 421 - blockidx_x  0 -  blockidx_y  3- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 422 - blockidx_x  0 -  blockidx_y  3- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 423 - blockidx_x  0 -  blockidx_y  3- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 424 - blockidx_x  0 -  blockidx_y  3- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 425 - blockidx_x  0 -  blockidx_y  3- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 426 - blockidx_x  0 -  blockidx_y  3- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 427 - blockidx_x  0 -  blockidx_y  3- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 428 - blockidx_x  0 -  blockidx_y  3- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 429 - blockidx_x  0 -  blockidx_y  3- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 430 - blockidx_x  0 -  blockidx_y  3- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 431 - blockidx_x  0 -  blockidx_y  3- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 432 - blockidx_x  0 -  blockidx_y  3- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 433 - blockidx_x  0 -  blockidx_y  3- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 434 - blockidx_x  0 -  blockidx_y  3- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 435 - blockidx_x  0 -  blockidx_y  3- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 436 - blockidx_x  0 -  blockidx_y  3- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 437 - blockidx_x  0 -  blockidx_y  3- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 438 - blockidx_x  0 -  blockidx_y  3- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 439 - blockidx_x  0 -  blockidx_y  3- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 440 - blockidx_x  0 -  blockidx_y  3- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 441 - blockidx_x  0 -  blockidx_y  3- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 442 - blockidx_x  0 -  blockidx_y  3- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 443 - blockidx_x  0 -  blockidx_y  3- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 444 - blockidx_x  0 -  blockidx_y  3- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 445 - blockidx_x  0 -  blockidx_y  3- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 446 - blockidx_x  0 -  blockidx_y  3- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 447 - blockidx_x  0 -  blockidx_y  3- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y 13 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 448 - blockidx_x  0 -  blockidx_y  3- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 449 - blockidx_x  0 -  blockidx_y  3- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 450 - blockidx_x  0 -  blockidx_y  3- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 451 - blockidx_x  0 -  blockidx_y  3- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 452 - blockidx_x  0 -  blockidx_y  3- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 453 - blockidx_x  0 -  blockidx_y  3- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 454 - blockidx_x  0 -  blockidx_y  3- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 455 - blockidx_x  0 -  blockidx_y  3- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 456 - blockidx_x  0 -  blockidx_y  3- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 457 - blockidx_x  0 -  blockidx_y  3- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 458 - blockidx_x  0 -  blockidx_y  3- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 459 - blockidx_x  0 -  blockidx_y  3- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 460 - blockidx_x  0 -  blockidx_y  3- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 461 - blockidx_x  0 -  blockidx_y  3- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 462 - blockidx_x  0 -  blockidx_y  3- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 463 - blockidx_x  0 -  blockidx_y  3- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 464 - blockidx_x  0 -  blockidx_y  3- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 465 - blockidx_x  0 -  blockidx_y  3- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 466 - blockidx_x  0 -  blockidx_y  3- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 467 - blockidx_x  0 -  blockidx_y  3- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 468 - blockidx_x  0 -  blockidx_y  3- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 469 - blockidx_x  0 -  blockidx_y  3- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 470 - blockidx_x  0 -  blockidx_y  3- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 471 - blockidx_x  0 -  blockidx_y  3- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 472 - blockidx_x  0 -  blockidx_y  3- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 473 - blockidx_x  0 -  blockidx_y  3- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 474 - blockidx_x  0 -  blockidx_y  3- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 475 - blockidx_x  0 -  blockidx_y  3- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 476 - blockidx_x  0 -  blockidx_y  3- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 477 - blockidx_x  0 -  blockidx_y  3- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 478 - blockidx_x  0 -  blockidx_y  3- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 479 - blockidx_x  0 -  blockidx_y  3- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 14 

graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 480 - blockidx_x  0 -  blockidx_y  3- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 481 - blockidx_x  0 -  blockidx_y  3- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 482 - blockidx_x  0 -  blockidx_y  3- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 483 - blockidx_x  0 -  blockidx_y  3- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 484 - blockidx_x  0 -  blockidx_y  3- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 485 - blockidx_x  0 -  blockidx_y  3- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 486 - blockidx_x  0 -  blockidx_y  3- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 487 - blockidx_x  0 -  blockidx_y  3- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 488 - blockidx_x  0 -  blockidx_y  3- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 489 - blockidx_x  0 -  blockidx_y  3- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 490 - blockidx_x  0 -  blockidx_y  3- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 491 - blockidx_x  0 -  blockidx_y  3- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 492 - blockidx_x  0 -  blockidx_y  3- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 493 - blockidx_x  0 -  blockidx_y  3- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 494 - blockidx_x  0 -  blockidx_y  3- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 495 - blockidx_x  0 -  blockidx_y  3- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 496 - blockidx_x  0 -  blockidx_y  3- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 497 - blockidx_x  0 -  blockidx_y  3- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 498 - blockidx_x  0 -  blockidx_y  3- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 499 - blockidx_x  0 -  blockidx_y  3- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 500 - blockidx_x  0 -  blockidx_y  3- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 501 - blockidx_x  0 -  blockidx_y  3- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 502 - blockidx_x  0 -  blockidx_y  3- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 503 - blockidx_x  0 -  blockidx_y  3- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 504 - blockidx_x  0 -  blockidx_y  3- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 505 - blockidx_x  0 -  blockidx_y  3- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 506 - blockidx_x  0 -  blockidx_y  3- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 507 - blockidx_x  0 -  blockidx_y  3- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 508 - blockidx_x  0 -  blockidx_y  3- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 509 - blockidx_x  0 -  blockidx_y  3- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 510 - blockidx_x  0 -  blockidx_y  3- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 511 - blockidx_x  0 -  blockidx_y  3- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 15 

```