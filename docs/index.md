# CUDATutorial

从零开始学习 CUDA 高性能编程，从入门到放弃，哦不！一起来边学习，边打笔记，日拱一卒！

![memory-hierarcy](./img/memory-hierarchy-in-gpus.png)

## 学习路线

### 新手村系列

+ [构建 CUDA 编程环境](/build_dev_env)
+ [手写第一个 Kernel](/first_kernel)
+ [nvprof 性能分析](/nvprof_usage)
+ [尝试第一次优化 Kernel](/first_refine_kernel)
+ [了解cuda线程分布](/what_my_id)


### 初阶系列

+ [初识多线程并行计算](/intro_parallel)
+ [手写实现矩阵乘 Matmul](/impl_matmul)
+ [矩阵乘 Matmul 性能优化实践](/optimize_matmul)

### 中阶系列

+ [手写实现 Reduce](/impl_reduce)
+ [Reduce 性能优化实践](/optimize_reduce)
+ [Reduce 性能优化实践—交叉寻址](/optimize_reduce/interleaved_addressing)
+ [Reduce 性能优化实践—解决 Bank Conflict](/optimize_reduce/bank_conflict)
+ [Reduce 性能优化实践—解决空闲线程](/optimize_reduce/idle_threads_free)
+ [Reduce 性能优化实践—展开最后一个 warp](/optimize_reduce/unroll)
+ [GEMM 优化专题-二维 Thread Tile 并行优化](/gemm_optimize/2dtiled)
+ [GEMM 优化专题-向量化 Shared Memory 和 Global Memory 访问](/gemm_optimize/vectorize_smem_and_gmem_accesses)
+ [GEMM 优化专题-warp tiling](/gemm_optimize/warptiling)
+ [GEMM 优化专题-双缓冲](/gemm_optimize/double_buffer)
+ [GEMM 优化专题-解决 Bank Conflict](/gemm_optimize/bank_conflicts)
+ [卷积算子优化专题-卷积算子简易实现](/convolution/naive_conv)
+ [卷积算子优化专题-卷积算子优化思路介绍](/convolution/intro_conv_optimize)
+ [卷积算子优化专题-im2col + gemm 实现卷积](/convolution/im2col_conv)
+ [卷积算子优化专题-隐式 GEMM 实现卷积](/convolution/implicit_gemm)
+ [卷积算子优化专题-CUTLASS 中的卷积优化策略](/convolution/cutlass_conv)




### 高阶系列

+ 页锁定和主机内存
+ CUDA 流和多流使用
+ 使用多个 GPU 计算
+ ...(补充中)

### 大师系列

我现在还不知道写啥，毕竟我现在还是菜鸡~~
