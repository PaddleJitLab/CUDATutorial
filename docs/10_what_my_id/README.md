# 打印线程号相关信息
<<<<<<< HEAD
本章节旨在帮助用户了解cuda内部线程块划分的规则，理解线程号的计算逻辑。

## 1. 1维block和1维thread
样例中设置了两个block,每个block中64个线程,  blockDim.x = 64
blockIdx.x 代表当前线程所在第几个block; 
threadIdx.x 代表当前现在在当前block中是第几个thread; 
warp_idx 代表当前线程在当前block中是第几个warp;（warp 会选择相邻的线程号做组合）
calc_idx 代表当前线程计算的是全局的第几个thread; 
block的索引 * 每个block的thread个数 + block内的thread索引 计算出全局索引。
=======
    本章节旨在帮助用户了解cuda内部线程块划分的规则，理解线程号的计算逻辑。

## 1. 1维block和1维thread
    样例中设置了两个block,每个block中64个线程,  blockDim.x = 64
    blockIdx.x 代表当前线程所在第几个block; 
    threadIdx.x 代表当前现在在当前block中是第几个thread; 
    warp_idx 代表当前线程在当前block中是第几个warp;（warp 会选择相邻的线程号做组合）
    calc_idx 代表当前线程计算的是全局的第几个thread; 
    block的索引 * 每个block的thread个数 + block内的thread索引 计算出全局索引。
>>>>>>> b47864458a8c315a0f6aace10ded33296bccb9e6
```c++
   const unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

<<<<<<< HEAD
编译命令 
```bash
nvcc my_id.cu -o my_id
```
执行命令
```bash
./my_id
```
运行结果
=======
>>>>>>> b47864458a8c315a0f6aace10ded33296bccb9e6
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
## 2.  2维block和2维thread
<<<<<<< HEAD
前四列是用户调用kernel时设置的block个数 4（1，4）thread个数 128（32, 4) 共计512个线程，每个线程中该数固定。

gradDim.x 描述block 在x维上的个数； gradDim.y 描述block 在y维上的个数； 
blcokDim.x 描述每个block 的x维上thread的个数；blockDim.y 描述每个block 的y维上thread的个数；

第五列代表当前线程计算的是全局的第几个thread;

第六到九列分别描述当前线程blockIdx.x: 在grid 的x维上第几个block, blockIdx.y: grid的y维上第几个块，
                        threadIdx.x: 在block 的x维上第几个thread，threadIdx.y: 在block的y维上第几thread,
=======
    前四列是用户调用kernel时设置的block个数 4（1，4）thread个数 128（32, 4) 共计512个线程，每个线程中该数固定。

    gradDim.x 描述block 在x维上的个数； gradDim.y 描述block 在y维上的个数； 
    blcokDim.x 描述每个block 的x维上thread的个数；blockDim.y 描述每个block 的y维上thread的个数；

    第五列代表当前线程计算的是全局的第几个thread;
    
    第六到九列分别描述当前线程blockIdx.x: 在grid 的x维上第几个block, blockIdx.y: grid的y维上第几个块，
                          threadIdx.x: 在block 的x维上第几个thread，threadIdx.y: 在block的y维上第几thread,
>>>>>>> b47864458a8c315a0f6aace10ded33296bccb9e6

```c++
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
```
    第十列，十一列计算了当前线程idx: 在grid 的x维上第几个thread, idy: 在grid 的y维上第几个thread, 

    由此可推导全局的索引 = 每行的thread 数 * 行数 + 单行的列偏移

```c++
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx ;
```
<<<<<<< HEAD
编译命令 
```bash
nvcc my_id_dim2.cu -o my_id_dim2
```
执行命令
```bash
./my_id_dim2
```
运行结果
=======

>>>>>>> b47864458a8c315a0f6aace10ded33296bccb9e6
```bash
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   0 - block_x  0 -  block_y  0- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   1 - block_x  0 -  block_y  0- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   2 - block_x  0 -  block_y  0- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   3 - block_x  0 -  block_y  0- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   4 - block_x  0 -  block_y  0- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   5 - block_x  0 -  block_y  0- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   6 - block_x  0 -  block_y  0- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   7 - block_x  0 -  block_y  0- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   8 - block_x  0 -  block_y  0- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread   9 - block_x  0 -  block_y  0- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  10 - block_x  0 -  block_y  0- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  11 - block_x  0 -  block_y  0- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  12 - block_x  0 -  block_y  0- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  13 - block_x  0 -  block_y  0- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  14 - block_x  0 -  block_y  0- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  15 - block_x  0 -  block_y  0- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  16 - block_x  0 -  block_y  0- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  17 - block_x  0 -  block_y  0- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  18 - block_x  0 -  block_y  0- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  19 - block_x  0 -  block_y  0- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  20 - block_x  0 -  block_y  0- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  21 - block_x  0 -  block_y  0- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  22 - block_x  0 -  block_y  0- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  23 - block_x  0 -  block_y  0- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  24 - block_x  0 -  block_y  0- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  25 - block_x  0 -  block_y  0- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  26 - block_x  0 -  block_y  0- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  27 - block_x  0 -  block_y  0- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  28 - block_x  0 -  block_y  0- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  29 - block_x  0 -  block_y  0- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  30 - block_x  0 -  block_y  0- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  31 - block_x  0 -  block_y  0- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  32 - block_x  0 -  block_y  0- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  33 - block_x  0 -  block_y  0- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  34 - block_x  0 -  block_y  0- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  35 - block_x  0 -  block_y  0- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  36 - block_x  0 -  block_y  0- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  37 - block_x  0 -  block_y  0- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  38 - block_x  0 -  block_y  0- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  39 - block_x  0 -  block_y  0- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  40 - block_x  0 -  block_y  0- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  41 - block_x  0 -  block_y  0- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  42 - block_x  0 -  block_y  0- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  43 - block_x  0 -  block_y  0- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  44 - block_x  0 -  block_y  0- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  45 - block_x  0 -  block_y  0- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  46 - block_x  0 -  block_y  0- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  47 - block_x  0 -  block_y  0- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  48 - block_x  0 -  block_y  0- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  49 - block_x  0 -  block_y  0- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  50 - block_x  0 -  block_y  0- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  51 - block_x  0 -  block_y  0- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  52 - block_x  0 -  block_y  0- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  53 - block_x  0 -  block_y  0- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  54 - block_x  0 -  block_y  0- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  55 - block_x  0 -  block_y  0- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  56 - block_x  0 -  block_y  0- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  57 - block_x  0 -  block_y  0- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  58 - block_x  0 -  block_y  0- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  59 - block_x  0 -  block_y  0- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  60 - block_x  0 -  block_y  0- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  61 - block_x  0 -  block_y  0- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  62 - block_x  0 -  block_y  0- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  63 - block_x  0 -  block_y  0- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  64 - block_x  0 -  block_y  0- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  65 - block_x  0 -  block_y  0- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  66 - block_x  0 -  block_y  0- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  67 - block_x  0 -  block_y  0- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  68 - block_x  0 -  block_y  0- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  69 - block_x  0 -  block_y  0- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  70 - block_x  0 -  block_y  0- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  71 - block_x  0 -  block_y  0- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  72 - block_x  0 -  block_y  0- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  73 - block_x  0 -  block_y  0- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  74 - block_x  0 -  block_y  0- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  75 - block_x  0 -  block_y  0- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  76 - block_x  0 -  block_y  0- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  77 - block_x  0 -  block_y  0- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  78 - block_x  0 -  block_y  0- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  79 - block_x  0 -  block_y  0- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  80 - block_x  0 -  block_y  0- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  81 - block_x  0 -  block_y  0- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  82 - block_x  0 -  block_y  0- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  83 - block_x  0 -  block_y  0- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  84 - block_x  0 -  block_y  0- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  85 - block_x  0 -  block_y  0- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  86 - block_x  0 -  block_y  0- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  87 - block_x  0 -  block_y  0- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  88 - block_x  0 -  block_y  0- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  89 - block_x  0 -  block_y  0- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  90 - block_x  0 -  block_y  0- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  91 - block_x  0 -  block_y  0- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  92 - block_x  0 -  block_y  0- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  93 - block_x  0 -  block_y  0- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  94 - block_x  0 -  block_y  0- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  95 - block_x  0 -  block_y  0- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  96 - block_x  0 -  block_y  0- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  97 - block_x  0 -  block_y  0- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  98 - block_x  0 -  block_y  0- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread  99 - block_x  0 -  block_y  0- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 100 - block_x  0 -  block_y  0- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 101 - block_x  0 -  block_y  0- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 102 - block_x  0 -  block_y  0- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 103 - block_x  0 -  block_y  0- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 104 - block_x  0 -  block_y  0- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 105 - block_x  0 -  block_y  0- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 106 - block_x  0 -  block_y  0- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 107 - block_x  0 -  block_y  0- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 108 - block_x  0 -  block_y  0- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 109 - block_x  0 -  block_y  0- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 110 - block_x  0 -  block_y  0- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 111 - block_x  0 -  block_y  0- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 112 - block_x  0 -  block_y  0- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 113 - block_x  0 -  block_y  0- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 114 - block_x  0 -  block_y  0- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 115 - block_x  0 -  block_y  0- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 116 - block_x  0 -  block_y  0- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 117 - block_x  0 -  block_y  0- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 118 - block_x  0 -  block_y  0- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 119 - block_x  0 -  block_y  0- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 120 - block_x  0 -  block_y  0- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 121 - block_x  0 -  block_y  0- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 122 - block_x  0 -  block_y  0- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 123 - block_x  0 -  block_y  0- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 124 - block_x  0 -  block_y  0- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 125 - block_x  0 -  block_y  0- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 126 - block_x  0 -  block_y  0- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 127 - block_x  0 -  block_y  0- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 128 - block_x  0 -  block_y  1- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 129 - block_x  0 -  block_y  1- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 130 - block_x  0 -  block_y  1- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 131 - block_x  0 -  block_y  1- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 132 - block_x  0 -  block_y  1- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 133 - block_x  0 -  block_y  1- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 134 - block_x  0 -  block_y  1- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 135 - block_x  0 -  block_y  1- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 136 - block_x  0 -  block_y  1- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 137 - block_x  0 -  block_y  1- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 138 - block_x  0 -  block_y  1- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 139 - block_x  0 -  block_y  1- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 140 - block_x  0 -  block_y  1- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 141 - block_x  0 -  block_y  1- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 142 - block_x  0 -  block_y  1- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 143 - block_x  0 -  block_y  1- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 144 - block_x  0 -  block_y  1- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 145 - block_x  0 -  block_y  1- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 146 - block_x  0 -  block_y  1- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 147 - block_x  0 -  block_y  1- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 148 - block_x  0 -  block_y  1- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 149 - block_x  0 -  block_y  1- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 150 - block_x  0 -  block_y  1- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 151 - block_x  0 -  block_y  1- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 152 - block_x  0 -  block_y  1- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 153 - block_x  0 -  block_y  1- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 154 - block_x  0 -  block_y  1- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 155 - block_x  0 -  block_y  1- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 156 - block_x  0 -  block_y  1- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 157 - block_x  0 -  block_y  1- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 158 - block_x  0 -  block_y  1- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 159 - block_x  0 -  block_y  1- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 160 - block_x  0 -  block_y  1- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 161 - block_x  0 -  block_y  1- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 162 - block_x  0 -  block_y  1- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 163 - block_x  0 -  block_y  1- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 164 - block_x  0 -  block_y  1- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 165 - block_x  0 -  block_y  1- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 166 - block_x  0 -  block_y  1- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 167 - block_x  0 -  block_y  1- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 168 - block_x  0 -  block_y  1- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 169 - block_x  0 -  block_y  1- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 170 - block_x  0 -  block_y  1- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 171 - block_x  0 -  block_y  1- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 172 - block_x  0 -  block_y  1- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 173 - block_x  0 -  block_y  1- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 174 - block_x  0 -  block_y  1- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 175 - block_x  0 -  block_y  1- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 176 - block_x  0 -  block_y  1- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 177 - block_x  0 -  block_y  1- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 178 - block_x  0 -  block_y  1- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 179 - block_x  0 -  block_y  1- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 180 - block_x  0 -  block_y  1- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 181 - block_x  0 -  block_y  1- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 182 - block_x  0 -  block_y  1- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 183 - block_x  0 -  block_y  1- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 184 - block_x  0 -  block_y  1- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 185 - block_x  0 -  block_y  1- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 186 - block_x  0 -  block_y  1- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 187 - block_x  0 -  block_y  1- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 188 - block_x  0 -  block_y  1- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 189 - block_x  0 -  block_y  1- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 190 - block_x  0 -  block_y  1- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 191 - block_x  0 -  block_y  1- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 192 - block_x  0 -  block_y  1- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 193 - block_x  0 -  block_y  1- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 194 - block_x  0 -  block_y  1- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 195 - block_x  0 -  block_y  1- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 196 - block_x  0 -  block_y  1- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 197 - block_x  0 -  block_y  1- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 198 - block_x  0 -  block_y  1- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 199 - block_x  0 -  block_y  1- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 200 - block_x  0 -  block_y  1- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 201 - block_x  0 -  block_y  1- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 202 - block_x  0 -  block_y  1- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 203 - block_x  0 -  block_y  1- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 204 - block_x  0 -  block_y  1- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 205 - block_x  0 -  block_y  1- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 206 - block_x  0 -  block_y  1- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 207 - block_x  0 -  block_y  1- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 208 - block_x  0 -  block_y  1- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 209 - block_x  0 -  block_y  1- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 210 - block_x  0 -  block_y  1- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 211 - block_x  0 -  block_y  1- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 212 - block_x  0 -  block_y  1- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 213 - block_x  0 -  block_y  1- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 214 - block_x  0 -  block_y  1- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 215 - block_x  0 -  block_y  1- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 216 - block_x  0 -  block_y  1- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 217 - block_x  0 -  block_y  1- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 218 - block_x  0 -  block_y  1- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 219 - block_x  0 -  block_y  1- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 220 - block_x  0 -  block_y  1- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 221 - block_x  0 -  block_y  1- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 222 - block_x  0 -  block_y  1- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 223 - block_x  0 -  block_y  1- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 224 - block_x  0 -  block_y  1- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 225 - block_x  0 -  block_y  1- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 226 - block_x  0 -  block_y  1- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 227 - block_x  0 -  block_y  1- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 228 - block_x  0 -  block_y  1- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 229 - block_x  0 -  block_y  1- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 230 - block_x  0 -  block_y  1- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 231 - block_x  0 -  block_y  1- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 232 - block_x  0 -  block_y  1- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 233 - block_x  0 -  block_y  1- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 234 - block_x  0 -  block_y  1- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 235 - block_x  0 -  block_y  1- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 236 - block_x  0 -  block_y  1- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 237 - block_x  0 -  block_y  1- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 238 - block_x  0 -  block_y  1- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 239 - block_x  0 -  block_y  1- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 240 - block_x  0 -  block_y  1- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 241 - block_x  0 -  block_y  1- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 242 - block_x  0 -  block_y  1- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 243 - block_x  0 -  block_y  1- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 244 - block_x  0 -  block_y  1- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 245 - block_x  0 -  block_y  1- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 246 - block_x  0 -  block_y  1- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 247 - block_x  0 -  block_y  1- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 248 - block_x  0 -  block_y  1- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 249 - block_x  0 -  block_y  1- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 250 - block_x  0 -  block_y  1- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 251 - block_x  0 -  block_y  1- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 252 - block_x  0 -  block_y  1- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 253 - block_x  0 -  block_y  1- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 254 - block_x  0 -  block_y  1- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 255 - block_x  0 -  block_y  1- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 256 - block_x  0 -  block_y  2- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 257 - block_x  0 -  block_y  2- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 258 - block_x  0 -  block_y  2- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 259 - block_x  0 -  block_y  2- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 260 - block_x  0 -  block_y  2- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 261 - block_x  0 -  block_y  2- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 262 - block_x  0 -  block_y  2- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 263 - block_x  0 -  block_y  2- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 264 - block_x  0 -  block_y  2- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 265 - block_x  0 -  block_y  2- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 266 - block_x  0 -  block_y  2- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 267 - block_x  0 -  block_y  2- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 268 - block_x  0 -  block_y  2- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 269 - block_x  0 -  block_y  2- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 270 - block_x  0 -  block_y  2- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 271 - block_x  0 -  block_y  2- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 272 - block_x  0 -  block_y  2- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 273 - block_x  0 -  block_y  2- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 274 - block_x  0 -  block_y  2- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 275 - block_x  0 -  block_y  2- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 276 - block_x  0 -  block_y  2- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 277 - block_x  0 -  block_y  2- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 278 - block_x  0 -  block_y  2- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 279 - block_x  0 -  block_y  2- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 280 - block_x  0 -  block_y  2- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 281 - block_x  0 -  block_y  2- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 282 - block_x  0 -  block_y  2- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 283 - block_x  0 -  block_y  2- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 284 - block_x  0 -  block_y  2- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 285 - block_x  0 -  block_y  2- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 286 - block_x  0 -  block_y  2- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 287 - block_x  0 -  block_y  2- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 288 - block_x  0 -  block_y  2- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 289 - block_x  0 -  block_y  2- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 290 - block_x  0 -  block_y  2- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 291 - block_x  0 -  block_y  2- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 292 - block_x  0 -  block_y  2- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 293 - block_x  0 -  block_y  2- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 294 - block_x  0 -  block_y  2- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 295 - block_x  0 -  block_y  2- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 296 - block_x  0 -  block_y  2- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 297 - block_x  0 -  block_y  2- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 298 - block_x  0 -  block_y  2- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 299 - block_x  0 -  block_y  2- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 300 - block_x  0 -  block_y  2- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 301 - block_x  0 -  block_y  2- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 302 - block_x  0 -  block_y  2- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 303 - block_x  0 -  block_y  2- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 304 - block_x  0 -  block_y  2- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 305 - block_x  0 -  block_y  2- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 306 - block_x  0 -  block_y  2- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 307 - block_x  0 -  block_y  2- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 308 - block_x  0 -  block_y  2- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 309 - block_x  0 -  block_y  2- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 310 - block_x  0 -  block_y  2- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 311 - block_x  0 -  block_y  2- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 312 - block_x  0 -  block_y  2- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 313 - block_x  0 -  block_y  2- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 314 - block_x  0 -  block_y  2- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 315 - block_x  0 -  block_y  2- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 316 - block_x  0 -  block_y  2- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 317 - block_x  0 -  block_y  2- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 318 - block_x  0 -  block_y  2- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 319 - block_x  0 -  block_y  2- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 320 - block_x  0 -  block_y  2- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 321 - block_x  0 -  block_y  2- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 322 - block_x  0 -  block_y  2- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 323 - block_x  0 -  block_y  2- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 324 - block_x  0 -  block_y  2- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 325 - block_x  0 -  block_y  2- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 326 - block_x  0 -  block_y  2- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 327 - block_x  0 -  block_y  2- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 328 - block_x  0 -  block_y  2- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 329 - block_x  0 -  block_y  2- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 330 - block_x  0 -  block_y  2- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 331 - block_x  0 -  block_y  2- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 332 - block_x  0 -  block_y  2- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 333 - block_x  0 -  block_y  2- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 334 - block_x  0 -  block_y  2- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 335 - block_x  0 -  block_y  2- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 336 - block_x  0 -  block_y  2- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 337 - block_x  0 -  block_y  2- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 338 - block_x  0 -  block_y  2- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 339 - block_x  0 -  block_y  2- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 340 - block_x  0 -  block_y  2- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 341 - block_x  0 -  block_y  2- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 342 - block_x  0 -  block_y  2- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 343 - block_x  0 -  block_y  2- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 344 - block_x  0 -  block_y  2- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 345 - block_x  0 -  block_y  2- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 346 - block_x  0 -  block_y  2- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 347 - block_x  0 -  block_y  2- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 348 - block_x  0 -  block_y  2- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 349 - block_x  0 -  block_y  2- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 350 - block_x  0 -  block_y  2- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 351 - block_x  0 -  block_y  2- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 352 - block_x  0 -  block_y  2- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 353 - block_x  0 -  block_y  2- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 354 - block_x  0 -  block_y  2- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 355 - block_x  0 -  block_y  2- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 356 - block_x  0 -  block_y  2- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 357 - block_x  0 -  block_y  2- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 358 - block_x  0 -  block_y  2- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 359 - block_x  0 -  block_y  2- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 360 - block_x  0 -  block_y  2- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 361 - block_x  0 -  block_y  2- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 362 - block_x  0 -  block_y  2- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 363 - block_x  0 -  block_y  2- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 364 - block_x  0 -  block_y  2- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 365 - block_x  0 -  block_y  2- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 366 - block_x  0 -  block_y  2- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 367 - block_x  0 -  block_y  2- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 368 - block_x  0 -  block_y  2- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 369 - block_x  0 -  block_y  2- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 370 - block_x  0 -  block_y  2- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 371 - block_x  0 -  block_y  2- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 372 - block_x  0 -  block_y  2- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 373 - block_x  0 -  block_y  2- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 374 - block_x  0 -  block_y  2- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 375 - block_x  0 -  block_y  2- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 376 - block_x  0 -  block_y  2- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 377 - block_x  0 -  block_y  2- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 378 - block_x  0 -  block_y  2- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 379 - block_x  0 -  block_y  2- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 380 - block_x  0 -  block_y  2- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 381 - block_x  0 -  block_y  2- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 382 - block_x  0 -  block_y  2- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 383 - block_x  0 -  block_y  2- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 384 - block_x  0 -  block_y  3- threadid_x  0 -  threadid_y  0 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 385 - block_x  0 -  block_y  3- threadid_x  1 -  threadid_y  0 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 386 - block_x  0 -  block_y  3- threadid_x  2 -  threadid_y  0 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 387 - block_x  0 -  block_y  3- threadid_x  3 -  threadid_y  0 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 388 - block_x  0 -  block_y  3- threadid_x  4 -  threadid_y  0 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 389 - block_x  0 -  block_y  3- threadid_x  5 -  threadid_y  0 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 390 - block_x  0 -  block_y  3- threadid_x  6 -  threadid_y  0 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 391 - block_x  0 -  block_y  3- threadid_x  7 -  threadid_y  0 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 392 - block_x  0 -  block_y  3- threadid_x  8 -  threadid_y  0 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 393 - block_x  0 -  block_y  3- threadid_x  9 -  threadid_y  0 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 394 - block_x  0 -  block_y  3- threadid_x 10 -  threadid_y  0 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 395 - block_x  0 -  block_y  3- threadid_x 11 -  threadid_y  0 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 396 - block_x  0 -  block_y  3- threadid_x 12 -  threadid_y  0 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 397 - block_x  0 -  block_y  3- threadid_x 13 -  threadid_y  0 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 398 - block_x  0 -  block_y  3- threadid_x 14 -  threadid_y  0 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 399 - block_x  0 -  block_y  3- threadid_x 15 -  threadid_y  0 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 400 - block_x  0 -  block_y  3- threadid_x 16 -  threadid_y  0 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 401 - block_x  0 -  block_y  3- threadid_x 17 -  threadid_y  0 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 402 - block_x  0 -  block_y  3- threadid_x 18 -  threadid_y  0 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 403 - block_x  0 -  block_y  3- threadid_x 19 -  threadid_y  0 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 404 - block_x  0 -  block_y  3- threadid_x 20 -  threadid_y  0 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 405 - block_x  0 -  block_y  3- threadid_x 21 -  threadid_y  0 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 406 - block_x  0 -  block_y  3- threadid_x 22 -  threadid_y  0 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 407 - block_x  0 -  block_y  3- threadid_x 23 -  threadid_y  0 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 408 - block_x  0 -  block_y  3- threadid_x 24 -  threadid_y  0 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 409 - block_x  0 -  block_y  3- threadid_x 25 -  threadid_y  0 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 410 - block_x  0 -  block_y  3- threadid_x 26 -  threadid_y  0 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 411 - block_x  0 -  block_y  3- threadid_x 27 -  threadid_y  0 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 412 - block_x  0 -  block_y  3- threadid_x 28 -  threadid_y  0 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 413 - block_x  0 -  block_y  3- threadid_x 29 -  threadid_y  0 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 414 - block_x  0 -  block_y  3- threadid_x 30 -  threadid_y  0 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 415 - block_x  0 -  block_y  3- threadid_x 31 -  threadid_y  0 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 416 - block_x  0 -  block_y  3- threadid_x  0 -  threadid_y  1 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 417 - block_x  0 -  block_y  3- threadid_x  1 -  threadid_y  1 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 418 - block_x  0 -  block_y  3- threadid_x  2 -  threadid_y  1 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 419 - block_x  0 -  block_y  3- threadid_x  3 -  threadid_y  1 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 420 - block_x  0 -  block_y  3- threadid_x  4 -  threadid_y  1 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 421 - block_x  0 -  block_y  3- threadid_x  5 -  threadid_y  1 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 422 - block_x  0 -  block_y  3- threadid_x  6 -  threadid_y  1 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 423 - block_x  0 -  block_y  3- threadid_x  7 -  threadid_y  1 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 424 - block_x  0 -  block_y  3- threadid_x  8 -  threadid_y  1 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 425 - block_x  0 -  block_y  3- threadid_x  9 -  threadid_y  1 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 426 - block_x  0 -  block_y  3- threadid_x 10 -  threadid_y  1 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 427 - block_x  0 -  block_y  3- threadid_x 11 -  threadid_y  1 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 428 - block_x  0 -  block_y  3- threadid_x 12 -  threadid_y  1 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 429 - block_x  0 -  block_y  3- threadid_x 13 -  threadid_y  1 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 430 - block_x  0 -  block_y  3- threadid_x 14 -  threadid_y  1 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 431 - block_x  0 -  block_y  3- threadid_x 15 -  threadid_y  1 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 432 - block_x  0 -  block_y  3- threadid_x 16 -  threadid_y  1 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 433 - block_x  0 -  block_y  3- threadid_x 17 -  threadid_y  1 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 434 - block_x  0 -  block_y  3- threadid_x 18 -  threadid_y  1 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 435 - block_x  0 -  block_y  3- threadid_x 19 -  threadid_y  1 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 436 - block_x  0 -  block_y  3- threadid_x 20 -  threadid_y  1 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 437 - block_x  0 -  block_y  3- threadid_x 21 -  threadid_y  1 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 438 - block_x  0 -  block_y  3- threadid_x 22 -  threadid_y  1 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 439 - block_x  0 -  block_y  3- threadid_x 23 -  threadid_y  1 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 440 - block_x  0 -  block_y  3- threadid_x 24 -  threadid_y  1 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 441 - block_x  0 -  block_y  3- threadid_x 25 -  threadid_y  1 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 442 - block_x  0 -  block_y  3- threadid_x 26 -  threadid_y  1 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 443 - block_x  0 -  block_y  3- threadid_x 27 -  threadid_y  1 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 444 - block_x  0 -  block_y  3- threadid_x 28 -  threadid_y  1 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 445 - block_x  0 -  block_y  3- threadid_x 29 -  threadid_y  1 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 446 - block_x  0 -  block_y  3- threadid_x 30 -  threadid_y  1 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 447 - block_x  0 -  block_y  3- threadid_x 31 -  threadid_y  1 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 448 - block_x  0 -  block_y  3- threadid_x  0 -  threadid_y  2 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 449 - block_x  0 -  block_y  3- threadid_x  1 -  threadid_y  2 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 450 - block_x  0 -  block_y  3- threadid_x  2 -  threadid_y  2 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 451 - block_x  0 -  block_y  3- threadid_x  3 -  threadid_y  2 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 452 - block_x  0 -  block_y  3- threadid_x  4 -  threadid_y  2 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 453 - block_x  0 -  block_y  3- threadid_x  5 -  threadid_y  2 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 454 - block_x  0 -  block_y  3- threadid_x  6 -  threadid_y  2 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 455 - block_x  0 -  block_y  3- threadid_x  7 -  threadid_y  2 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 456 - block_x  0 -  block_y  3- threadid_x  8 -  threadid_y  2 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 457 - block_x  0 -  block_y  3- threadid_x  9 -  threadid_y  2 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 458 - block_x  0 -  block_y  3- threadid_x 10 -  threadid_y  2 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 459 - block_x  0 -  block_y  3- threadid_x 11 -  threadid_y  2 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 460 - block_x  0 -  block_y  3- threadid_x 12 -  threadid_y  2 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 461 - block_x  0 -  block_y  3- threadid_x 13 -  threadid_y  2 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 462 - block_x  0 -  block_y  3- threadid_x 14 -  threadid_y  2 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 463 - block_x  0 -  block_y  3- threadid_x 15 -  threadid_y  2 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 464 - block_x  0 -  block_y  3- threadid_x 16 -  threadid_y  2 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 465 - block_x  0 -  block_y  3- threadid_x 17 -  threadid_y  2 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 466 - block_x  0 -  block_y  3- threadid_x 18 -  threadid_y  2 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 467 - block_x  0 -  block_y  3- threadid_x 19 -  threadid_y  2 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 468 - block_x  0 -  block_y  3- threadid_x 20 -  threadid_y  2 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 469 - block_x  0 -  block_y  3- threadid_x 21 -  threadid_y  2 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 470 - block_x  0 -  block_y  3- threadid_x 22 -  threadid_y  2 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 471 - block_x  0 -  block_y  3- threadid_x 23 -  threadid_y  2 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 472 - block_x  0 -  block_y  3- threadid_x 24 -  threadid_y  2 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 473 - block_x  0 -  block_y  3- threadid_x 25 -  threadid_y  2 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 474 - block_x  0 -  block_y  3- threadid_x 26 -  threadid_y  2 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 475 - block_x  0 -  block_y  3- threadid_x 27 -  threadid_y  2 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 476 - block_x  0 -  block_y  3- threadid_x 28 -  threadid_y  2 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 477 - block_x  0 -  block_y  3- threadid_x 29 -  threadid_y  2 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 478 - block_x  0 -  block_y  3- threadid_x 30 -  threadid_y  2 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 479 - block_x  0 -  block_y  3- threadid_x 31 -  threadid_y  2 - thread_x 31 - thread_y 31 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 480 - block_x  0 -  block_y  3- threadid_x  0 -  threadid_y  3 - thread_x  0 - thread_y  0 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 481 - block_x  0 -  block_y  3- threadid_x  1 -  threadid_y  3 - thread_x  1 - thread_y  1 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 482 - block_x  0 -  block_y  3- threadid_x  2 -  threadid_y  3 - thread_x  2 - thread_y  2 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 483 - block_x  0 -  block_y  3- threadid_x  3 -  threadid_y  3 - thread_x  3 - thread_y  3 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 484 - block_x  0 -  block_y  3- threadid_x  4 -  threadid_y  3 - thread_x  4 - thread_y  4 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 485 - block_x  0 -  block_y  3- threadid_x  5 -  threadid_y  3 - thread_x  5 - thread_y  5 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 486 - block_x  0 -  block_y  3- threadid_x  6 -  threadid_y  3 - thread_x  6 - thread_y  6 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 487 - block_x  0 -  block_y  3- threadid_x  7 -  threadid_y  3 - thread_x  7 - thread_y  7 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 488 - block_x  0 -  block_y  3- threadid_x  8 -  threadid_y  3 - thread_x  8 - thread_y  8 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 489 - block_x  0 -  block_y  3- threadid_x  9 -  threadid_y  3 - thread_x  9 - thread_y  9 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 490 - block_x  0 -  block_y  3- threadid_x 10 -  threadid_y  3 - thread_x 10 - thread_y 10 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 491 - block_x  0 -  block_y  3- threadid_x 11 -  threadid_y  3 - thread_x 11 - thread_y 11 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 492 - block_x  0 -  block_y  3- threadid_x 12 -  threadid_y  3 - thread_x 12 - thread_y 12 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 493 - block_x  0 -  block_y  3- threadid_x 13 -  threadid_y  3 - thread_x 13 - thread_y 13 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 494 - block_x  0 -  block_y  3- threadid_x 14 -  threadid_y  3 - thread_x 14 - thread_y 14 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 495 - block_x  0 -  block_y  3- threadid_x 15 -  threadid_y  3 - thread_x 15 - thread_y 15 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 496 - block_x  0 -  block_y  3- threadid_x 16 -  threadid_y  3 - thread_x 16 - thread_y 16 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 497 - block_x  0 -  block_y  3- threadid_x 17 -  threadid_y  3 - thread_x 17 - thread_y 17 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 498 - block_x  0 -  block_y  3- threadid_x 18 -  threadid_y  3 - thread_x 18 - thread_y 18 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 499 - block_x  0 -  block_y  3- threadid_x 19 -  threadid_y  3 - thread_x 19 - thread_y 19 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 500 - block_x  0 -  block_y  3- threadid_x 20 -  threadid_y  3 - thread_x 20 - thread_y 20 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 501 - block_x  0 -  block_y  3- threadid_x 21 -  threadid_y  3 - thread_x 21 - thread_y 21 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 502 - block_x  0 -  block_y  3- threadid_x 22 -  threadid_y  3 - thread_x 22 - thread_y 22 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 503 - block_x  0 -  block_y  3- threadid_x 23 -  threadid_y  3 - thread_x 23 - thread_y 23 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 504 - block_x  0 -  block_y  3- threadid_x 24 -  threadid_y  3 - thread_x 24 - thread_y 24 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 505 - block_x  0 -  block_y  3- threadid_x 25 -  threadid_y  3 - thread_x 25 - thread_y 25 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 506 - block_x  0 -  block_y  3- threadid_x 26 -  threadid_y  3 - thread_x 26 - thread_y 26 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 507 - block_x  0 -  block_y  3- threadid_x 27 -  threadid_y  3 - thread_x 27 - thread_y 27 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 508 - block_x  0 -  block_y  3- threadid_x 28 -  threadid_y  3 - thread_x 28 - thread_y 28 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 509 - block_x  0 -  block_y  3- threadid_x 29 -  threadid_y  3 - thread_x 29 - thread_y 29 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 510 - block_x  0 -  block_y  3- threadid_x 30 -  threadid_y  3 - thread_x 30 - thread_y 30 
graddim_x  1 - graddim_y  4 - blockdim_x 32 - blockdim_y  4 -cac_thread 511 - block_x  0 -  block_y  3- threadid_x 31 -  threadid_y  3 - thread_x 31 - thread_y 31 

```