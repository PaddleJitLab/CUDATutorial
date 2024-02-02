# CUDATutorial

从零开始学习 CUDA 高性能编程，从入门到放弃，哦不！一起来边学习，边打笔记，日拱一卒！

![memory-hierarcy](./img/memory-hierarchy-in-gpus.png)

## 学习路线

### 新手村系列

+ [构建 CUDA 编程环境](/build_dev_env)
+ [手写第一个 Kernel](/first_kernel)
+ [nvprof 性能分析](/nvprof_usage)
+ [尝试第一次优化 Kernel](/first_refine_kernel)
<<<<<<< HEAD
=======
+ [打印线程号相关信息](/what_my_id)
>>>>>>> 01e3e6171c623c6df27c92c6f9536b56997840ac


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

### 高阶系列

+ 页锁定和主机内存
+ CUDA 流和多流使用
+ 使用多个 GPU 计算
+ ...(补充中)

### 大师系列
我现在还不知道写啥，毕竟我现在还是菜鸡~~

