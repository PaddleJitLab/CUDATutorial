# CUDATutorial

从零开始学习 CUDA 高性能编程，从入门到放弃，哦不！一起来边学习，边打笔记，日拱一卒！
![memory-hierarcy](./img/memory-hierarchy-in-gpus.png)

## 学习路线

### 新手村系列

+ [构建 CUDA 编程环境](./01_build_dev_env/)
+ [手写第一个 Kernel](./02_first_kernel/)
+ [nvprof 性能分析](./03_nvprof_usage/)
+ [尝试第一次优化 Kernel](./04_first_refine_kernel/)


### 初阶系列

+ [初识多线程并行计算](./05_intro_parallel/)
+ [手写实现矩阵乘 Matmul](./06_impl_matmul/)
+ [矩阵乘 Matmul 性能优化实践](./07_optimize_matmul/)

### 中阶系列

+ [手写实现 Reduce](./08_impl_reduce/)
+ [Reduce 性能优化实践—交叉寻址](./09_optimize_reduce/01_interleaved_addressing/README.md)
+ [Reduce 性能优化实践—解决 Bank Conflict](./09_optimize_reduce/02_bank_conflict/README.md)
+ [Reduce 性能优化实践—解决 Idle 线程](./09_optimize_reduce/03_idle_thread/README.md)
+ [Reduce 性能优化实践—展开最后一个 warp](./09_optimize_reduce/04_unroll_last_warp/README.md)

### 高阶系列

+ 页锁定和主机内存
+ CUDA 流和多流使用
+ 使用多个 GPU 计算
+ ...(补充中)

### 大师系列

我现在还不知道写啥，毕竟我现在还是菜鸡~~