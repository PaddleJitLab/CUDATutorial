# 向量化访存

向量化访存是指将多个内存访问操作合并为一个内存访问操作。这样可以减少内存访问的次数，提高内存访问的效率。在本节中，我们将介绍如何通过向量化访存来提高矩阵乘法的性能。

## 1. 优化思路

上一个 Kernel 中加载矩阵 A 共享内存的代码如下:

```cpp
for (uint load_offset = 0; load_offset < BM; load_offset += stride_A)
{
    smem_A[(inner_row_A + load_offset) * BK + inner_col_A] = A[(inner_row_A + load_offset) * K + inner_col_A];
}
```

可以看到每次从 A 中读取一个元素, 且每次读取的元素不是连续的。我们可以使用向量读取指令 LDS.128 优化 Shared Memory 访问（对应 float4 数据类型）可以提高访存效率。

:::note

LDS.128指令可以一次性读取4个float类型的数据。

:::

同样在将结果写回到全局内存时，也可以一次型写入4个float类型的数据。这样可以减少全局内存访问的次数。

但是想要使用LDS.128指令，需要保证数据的访问是连续的。B 矩阵在内存中是按行存储的，因此在读取 B 矩阵的数据时，需要保证每个线程读取的数据是连续的。但是我们在读取 A 矩阵的数据时，是按列读取的，因此我们需要将 A 矩阵存入 smem_A 之前做一次转置。之前 smem_A 的大小是 BM * BK，现在我们将其转置为 BK * BM。

算法整体流程如下：

![picture 0](images/05eee538f6394ffc2ffffc2947edc8c888175af7152a150d697bfefb47db7a98.png)  

