# CUDATutorial

从零开始学习 CUDA 高性能编程，从入门到放弃，哦不！一起来边学习，边打笔记，日拱一卒！

> [!NOTE]
> 你可以访问 https://cuda.keter.top/ 来访问本仓库的网页版

![memory-hierarcy](./img/memory-hierarchy-in-gpus.png)

## 学习路线

### 新手村系列

+ [构建 CUDA 编程环境](./docs/01_build_dev_env/)
+ [手写第一个 Kernel](./docs/02_first_kernel/)
+ [nvprof 性能分析](./docs/03_nvprof_usage/)
+ [尝试第一次优化 Kernel](./docs/04_first_refine_kernel/)
+ [了解cuda线程分布](./docs/10_what_my_id/)

### 初阶系列

+ [初识多线程并行计算](./docs/05_intro_parallel/)
+ [手写实现矩阵乘 Matmul](./docs/06_impl_matmul/)
+ [矩阵乘 Matmul 性能优化实践](./docs/07_optimize_matmul/)
+ [打印线程号相关信息](./docs/10_what_my_id/)

### 中阶系列

+ [手写实现 Reduce](./docs/08_impl_reduce/)
+ [Reduce 性能优化实践—交叉寻址](./docs/09_optimize_reduce/01_interleaved_addressing/README.md)
+ [Reduce 性能优化实践—解决 Bank Conflict](./docs/09_optimize_reduce/02_bank_conflict/README.md)
+ [Reduce 性能优化实践—解决空闲线程](./docs/09_optimize_reduce/03_idle_threads_free/README.md)
+ [Reduce 性能优化实践—展开最后一个 warp](./docs/09_optimize_reduce/04_unroll/README.md)

### 高阶系列

+ 页锁定和主机内存
+ CUDA 流和多流使用
+ 使用多个 GPU 计算
+ ...(补充中)

### 大师系列

我现在还不知道写啥，毕竟我现在还是菜鸡~~
