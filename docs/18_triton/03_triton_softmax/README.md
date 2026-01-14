# Triton Softmax 算子实现

## 前言

前两篇文章咱们学习了 Triton 的编程范式和内存管理，今天来看一个更实用的算子：**Softmax**。

Softmax 是 Transformer 架构的核心组件，Attention 机制里必用。更重要的是，实现 Softmax 需要用到 Triton 中非常重要的 **Reduction 操作**，这是前两篇文章还没覆盖的内容。

通过这篇文章，你将学会如何用 Triton 实现数值稳定的 Softmax，也会看到 Triton 在处理 Reduction 操作时相比 CUDA 有多简洁。

## 一、从 Softmax 说起

### 1.1 数学回顾

Softmax 的公式大家都很熟悉：

$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

但直接这样实现会有数值问题：当 $x_i$ 很大时，$\exp(x_i)$ 可能会上溢出。所以工程上通常使用**数值稳定的版本**：

$$
\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}
$$

先减去最大值，这样指数的最大值是 $\exp(0) = 1$，不会溢出。

### 1.2 任务描述

给定一个 M×N 的矩阵 `x`，对**每一行**独立做 Softmax。也就是说：

- 输入：`x`，shape `[M, N]`
- 输出：`out`，shape `[M, N]`，每行元素和为 1

这是 Attention 机制中最常用的场景（M 个 query，N 个 key）。

## 二、CUDA 实现：手写 Reduction

在 CUDA 中实现 Softmax，最复杂的地方是 **Reduction** 操作——求每行的最大值和和。

### 2.1 Reduction 回顾

还记得咱们在[之前教程](https://cuda.keter.top/impl_reduce/)中学过的 Reduction 吗？在 CUDA 中求一个数组的最大值，需要：

```cpp
// 加载数据到 shared memory
__shared__ float sdata[BLOCKSIZE];
sdata[tid] = data[i];
__syncthreads();

// 多轮归约
for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
}
```

每次迭代，参与比较的线程数减半，最后 `sdata[0]` 就是最大值。Shared Memory + 多轮循环 + 同步，一个都不能少。

### 2.2 CUDA Softmax 完整实现

基于这个 Reduction 模式，咱们来实现 Softmax：

```cpp
template <int BLOCKSIZE>
__global__ void softmax_kernel_cuda(float *input, float *output, int rows, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= rows) return;

    // Shared Memory 用于 reduction
    __shared__ float s_max[BLOCKSIZE];
    __shared__ float s_sum[BLOCKSIZE];

    // === 第一步：求最大值 ===
    int idx = bid * cols + tid;
    float val = (tid < cols) ? input[idx] : -INFINITY;
    s_max[tid] = val;
    __syncthreads();

    for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float row_max = s_max[0];

    // === 第二步：计算 exp(x - max) 并求和 ===
    val = (tid < cols) ? expf(val - row_max) : 0.0f;
    s_sum[tid] = val;
    __syncthreads();

    for (int s = BLOCKSIZE / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < cols) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float row_sum = s_sum[0];

    // === 第三步：归一化并写回 ===
    if (tid < cols) {
        output[idx] = val / row_sum;
    }
}
```

:::tip

这个实现假设 `cols <= BLOCKSIZE`，即一个 block 能容纳一整行。如果 `cols` 更大，需要更复杂的多阶段 reduction。

:::

代码体现了 CUDA 编程的几个核心要点：

1. **Shared Memory 管理**：需要手动分配 `s_max` 和 `s_sum` 两个数组
2. **同步点**：每次 reduction 后都要 `__syncthreads()`，确保所有线程都完成
3. **边界处理**：`tid < cols` 检查，避免越界访问
4. **两次独立 Reduction**：先求 max，再求 sum，逻辑不重叠

softmax 的 cuda 源码可以在 [codes](https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs/18_triton/03_triton_softmax/codes) 目录下通过一下命令编译和执行：

```plain
make
./softmax
```

代码逻辑不算复杂，这里就不着重介绍了。

## 三、Triton 实现：一行代码的 Reduction

在 Triton 中，Reduction 变得异常简单。咱们来看看怎么实现。

### 3.1 Triton Reduction 的两种模式

在开始写代码之前，先理解一下 Triton 中 Reduction 的两种情况。

**模式 1：Program 内部 Reduction**

```python
x = tl.load(...)  # 加载一批数据
x_max = tl.max(x, axis=0)  # 在这批数据内求最大值
```

这是最常见的情况——单个 Program 处理的数据块自己内部归约。

当 Triton 看到 `tl.max(x, axis=0)` 时，它会生成类似 CUDA 的 reduction 循环，但细节由编译器自动处理：

```ptx
// Triton 生成的 PTX（简化版）
// 假设 BLOCK_SIZE = 512

// 1. 加载数据到寄存器
ld.global.f32 %r[0:511], [%ptr];

// 2. 使用 warp-level primitives 做 reduction
// 每个 warp（32 线程）内部先归约
red.max.sync.alignment.aligned.u32 %r_warp_max, %r[0:31];

// 3. 如果跨 warp，继续归约
// 最终得到单个标量值
```

Triton 会自动选择最高效的实现方式。对于小规模数据，可能直接用寄存器；对于大规模数据，会利用 shared memory + warp shuffle。

**模式 2：跨 Program Reduction**

```python
# 需要 tl.reduce 或 atomic operations
result = tl.reduce(x, op=tl.MAX, axis=0)
```

这种情况用于需要合并多个 Program 结果的场景，通常需要原子操作。

跨 Program reduction 的 PTX 会完全不同：

```ptx
// 跨 Program reduction（简化版）
// 每个 Program 先得到自己的局部结果
local_max = ...;

// 然后通过原子操作合并到全局结果
atom.max.s32 [%global_result], local_max;
```

或者使用多阶段 kernel：第一个 kernel 计算局部结果，第二个 kernel 合并。

**关键区别**：

| 特性 | Program 内部 Reduction | 跨 Program Reduction |
|------|----------------------|---------------------|
| API | `tl.max(x, axis=0)` | `tl.reduce(x, op=tl.MAX, axis=0)` |
| 数据范围 | 单个 Program 内部 | 跨越多个 Program |
| 同步需求 | 无需显式同步 | 可能需要原子操作或多 kernel |
| 性能 | 高（数据在寄存器/Shared Memory） | 相对较低（需要全局内存同步） |
| PTX 特征 | 使用 `red.*.sync` 指令 | 使用 `atom.*` 指令 |

对于 Softmax，咱们只需要**模式 1**：每个 Program 处理完整的一行，在行内做 Reduction。

### 3.2 核心思路

咱们的设计思路很清晰：

1. 每个 Program 处理**一整行**
2. 在 Program 内用 `tl.max()` 和 `tl.sum()` 做 Reduction
3. 前提：BLOCK_SIZE >= 列数（单 block 能容纳整行）

这意味着启动 M 个 Program（M 是行数），每个 Program 独立完成一行的 Softmax，互不干扰。

### 3.3 Kernel 函数定义

先写函数签名：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,          # 输入指针 [M, N]
    output_ptr,     # 输出指针 [M, N]
    n_rows,         # 行数 M
    n_cols,         # 列数 N
    BLOCK_SIZE: tl.constexpr,  # 块大小（必须 >= n_cols）
):
    """
    行级 Softmax Kernel：对输入矩阵的每一行独立做 softmax
    """
    pass
```

参数说明：
- `x_ptr`, `output_ptr`：输入输出指针
- `n_rows`, `n_cols`：矩阵维度
- `BLOCK_SIZE`：编译时常量，必须是 2 的幂次

### 3.4 完整实现

现在咱们一段一段来填充 kernel 的主体。

首先，每个 Program 处理一行，所以需要获取行号和列偏移：

```python
@triton.jit
def softmax_kernel(
    x_ptr, output_ptr, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行
    row_idx = tl.program_id(axis=0)

    # 计算该行的列偏移量（向量化）
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 计算行首地址
    row_start = x_ptr + row_idx * n_cols
```

这里 `row_idx` 是当前 Program 负责的行号，`n_cols` 是行 stride（因为矩阵是行优先存储的）。

接下来处理边界情况并加载数据：

```python
    # 创建 mask：处理列数不是 BLOCK_SIZE 倍数的情况
    mask = col_offsets < n_cols

    # 加载一行数据
    # other=-float('inf')：mask=False 的位置用 -inf 填充
    x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
```

:::note

**为什么用 `-inf`？**
当 mask 为 False 时，对应位置填充 -inf。这样：
- 求 max 时：-inf 不会影响结果（任何数 > -inf）
- 求 exp 时：exp(-inf) = 0，不影响 sum

:::

这比 CUDA 的 `if (tid < cols)` 更优雅，不会引起分支分歧。

到了最核心的部分——Reduction。在 Triton 中，只需要一行代码：

```python
    # === 数值稳定的 Softmax ===
    # 1. 求行内最大值（用于数值稳定性）
    x_max = tl.max(x, axis=0)

    # 2. 减去最大值后计算指数
    x_exp = tl.exp(x - x_max)

    # 3. 求指数和
    x_sum = tl.sum(x_exp, axis=0)

    # 4. 归一化
    output = x_exp / x_sum
```

对比 CUDA 的 reduction 循环，Triton 的 `tl.max(x, axis=0)` 一行搞定。`axis=0` 表示沿着 `col_offsets` 维度归约（即行内归约）。

最后把结果写回去：

```python
    # 写回结果
    out_row_start = output_ptr + row_idx * n_cols
    tl.store(out_row_start + col_offsets, output, mask=mask)
```

完整的 kernel 就这么几十行，比 CUDA 版本简洁很多。

现在写一个 Python 函数来调用这个 kernel：

```python
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton 实现的行级 Softmax

    Args:
        x: 输入张量，shape [M, N]

    Returns:
        输出张量，shape [M, N]，每行元素和为 1
    """
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # 设置 block size：必须是 2 的幂次，且 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 启动 kernel：n_rows 个 program，每个处理一行
    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
```

使用示例：

```python
if __name__ == "__main__":
    # 创建测试数据
    torch.manual_seed(0)
    x = torch.randn(1024, 128, device='cuda')

    # Triton 结果
    y_triton = softmax(x)

    # PyTorch 结果（作为参考）
    y_torch = torch.nn.functional.softmax(x, dim=-1)

    # 验证正确性
    print(f"Max error: {torch.max(torch.abs(y_triton - y_torch))}")
    assert torch.allclose(y_triton, y_torch, atol=1e-4)
    print("✓ Correctness check passed!")
```

完整的代码请参考 [`codes/softmax.py`](https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs/18_triton/03_triton_softmax/codes/softmax.py)。

## 四、代码详解与对比

### 4.1 Triton Reduction 的原理

你可能好奇，`tl.max(x, axis=0)` 背后是怎么工作的？

Triton 编译器会将这个操作编译成类似 CUDA 的 reduction 循环，但细节由编译器处理。对于 `x` 是一个向量（shape `[BLOCK_SIZE]`）的情况：

```python
x = tl.load(...)  # shape: [BLOCK_SIZE]
x_max = tl.max(x, axis=0)  # scalar
```

Triton 会生成类似下面的 PTX 代码：

```ptx
// 加载数据到寄存器
ld.global.b32 %r1, [ %rd1 + 0 ];

// Warp-level reduction: 使用 butterfly shuffle 模式
shfl.sync.bfly.b32 %r12, %r1, 16, 31, -1;  // 与 lane+16 交换
max.f32         %r13, %r1, %r12;             // 取最大值

shfl.sync.bfly.b32 %r14, %r13, 8, 31, -1;   // 与 lane+8 交换
max.f32         %r15, %r13, %r14;

shfl.sync.bfly.b32 %r16, %r15, 4, 31, -1;   // 与 lane+4 交换
max.f32         %r17, %r15, %r16;

shfl.sync.bfly.b32 %r18, %r17, 2, 31, -1;   // 与 lane+2 交换
max.f32         %r19, %r17, %r18;

shfl.sync.bfly.b32 %r20, %r19, 1, 31, -1;   // 与 lane+1 交换
max.f32         %r3, %r19, %r20;             // warp 内最大值

// 如果 BLOCK_SIZE > 32，需要跨 warp reduction
// 使用 shared memory 存储每个 warp 的结果
st.shared.b32 [ %r2 + 0 ], %r3;
bar.sync        0;                           // 同步所有 warp

// 前 4 个线程读取 shared memory 继续归约
ld.shared.b32 %r4, [ %r5 + 0 ];
shfl.sync.bfly.b32 %r25, %r4, 2, 31, -1;
max.f32         %r26, %r4, %r25;
shfl.sync.bfly.b32 %r27, %r26, 1, 31, -1;
max.f32         %r7, %r26, %r27;
```

这段 PTX 展示了 Triton 编译器如何实现 reduction：

1. **Warp 内 Butterfly Shuffle**：利用 `shfl.sync.bfly.b32` 指令，在 warp 内 32 个线程间进行 butterfly 模式的数据交换和 `max.f32` 比较，log₂(32)=5 轮即可完成 warp 内 reduction
2. **跨 Warp Reduction**：当 `BLOCK_SIZE > 32` 时，将每个 warp 的结果写入 shared memory，然后通过 barrier 同步，再用前几个线程从 shared memory 读取继续归约

这一切对你来说是无需感知的，你只需要写 `tl.max(x, axis=0)`，编译器会自动生成最优的 PTX 代码。

### 4.2 CUDA vs Triton 对比

| 操作 | CUDA | Triton |
|------|------|--------|
| Reduction | 手写 shared memory 循环 | `tl.max(x, axis=0)` |
| 同步 | 需要 `__syncthreads()` | 自动处理 |
| 边界处理 | `if (tid < cols)` | `mask=col_offsets < n_cols` |
| 代码量 | ~50 行 | ~30 行 |

### 4.3 Mask 机制的优势

CUDA 的 `if (tid < cols)` 会导致 Warp Divergence：

```cpp
// CUDA：同一个 warp 内可能走不同分支
if (tid < cols) {
    // 前面的线程执行这里
} else {
    // 后面的线程执行这里（可能被过滤掉）
}
```

Triton 的 mask 是**向量化**的：

```python
mask = col_offsets < n_cols  # 布尔向量
x = tl.load(..., mask=mask)  # 只加载有效位置
```

Triton 利用 Predicated Instructions（谓词指令）来屏蔽无效计算，避免了显式的控制流跳转，从而保持 Warp 内线程的高度同步和高效率。

:::note

**什么是带谓词的指令？**

谓词（predicate）是一种特殊的寄存器，存储 `True` 或 `False`。带谓词的指令是指：指令的执行与否取决于谓词的值。

在 PTX 中，这对应如下形式：

```ptx
// 假设 %p 是谓词寄存器（mask 的结果）
setp.lt.u32 %p, %col_offset, %n_cols;  // 比较并设置谓词

// 带谓词的加载指令
// 如果 %p 为 True，执行加载；如果为 False，跳过
ld.global.f32 {%r}, [%ptr], %p;
```

在带谓词的指令中，warp 内的所有线程**始终步调一致**，只是：

- 谓词为 True 的线程：真正执行加载并写回结果
- 谓词为 False 的线程：被禁用（不执行写回，但也不影响其他线程）

这样就不会产生 Warp Divergence，效率更高。

:::

完整的代码和正确性验证请参考 [`codes/softmax.py`](./codes/softmax.py)。

## 五、局限性与进阶方向

### 5.1 当前列数的限制

当前实现要求 `BLOCK_SIZE >= n_cols`，这意味着列数不能太大。这是为什么呢？

**根本原因：寄存器压力**

每个 Program 需要加载整行数据到寄存器：
```python
x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))  # shape: [BLOCK_SIZE]
```

当 `BLOCK_SIZE` 很大时（比如 4096），这会消耗大量寄存器：
- 现代 GPU（如 A100）每个 SM 大约有 256KB 寄存器
- 如果每个 float32 占 4 字节，4096 个元素就需要 16KB
- 考虑到中间变量（`x_max`, `x_exp`, `x_sum`），寄存器需求会翻倍

这会导致 **Occupancy 下降**：同一个 SM 上能同时驻留的 Program 数量减少，影响整体性能。

### 5.2 性能分析

咱们可以用 `triton.testing.do_bench` 来测量实际性能：

```python
import triton.testing

def benchmark(M, N):
    x = torch.randn(M, N, device='cuda')

    # 预热
    for _ in range(10):
        softmax(x)
    torch.cuda.synchronize()

    # 计时
    time_ms = triton.testing.do_bench(lambda: softmax(x), rep=100)

    # 带宽计算：读取 + 写入 = 2 * M * N * 4 bytes
    bandwidth_achieved = (2 * M * N * 4) / (time_ms * 1e-3) / 1e9  # GB/s

    # 获取GPU理论峰值带宽（这里假设是 H20 GPU）
    theoretical_bandwidth = 4096

    bandwidth_utilization = (bandwidth_achieved / theoretical_bandwidth) * 100

    return time_ms, bandwidth_achieved, bandwidth_utilization

def run_comprehensive_benchmark():
    print("=" * 60)
    print("Softmax 性能基准测试")
    print("=" * 60)

    test_cases = [
        # (M, N)
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (1024, 4096),
        (1024, 8192),
        (1024, 16384),
        (1024, 32768),
    ]

    print(f"{'配置':<20} {'时间(ms)':<12} {'带宽(GB/s)':<15} {'利用率(%)':<12} {'性能分析'}")
    print("-" * 80)

    for M, N in test_cases:
        time_ms, bandwidth, utilization = benchmark(M, N)

        # 性能分析
        if utilization > 70:
            analysis = "优秀 - 高效利用带宽"
        elif utilization > 50:
            analysis = "良好 - 带宽利用合理"
        elif utilization > 30:
            analysis = "一般 - 有优化空间"
        else:
            analysis = "较差 - 可能存在瓶颈"

        print(f"{M}×{N}".ljust(20), end="")
        print(f"{time_ms:.3f}".ljust(12), end="")
        print(f"{bandwidth:.1f}".ljust(15), end="")
        print(f"{utilization:.1f}%".ljust(12), end="")
        print(analysis)

    print("\n" + "=" * 60)
```

输出如下：

```plain
============================================================
Softmax 性能基准测试
============================================================
配置                   时间(ms)       带宽(GB/s)        利用率(%)       性能分析
--------------------------------------------------------------------------------
1024×512            0.007       582.0          14.2%       较差 - 可能存在瓶颈
1024×1024           0.008       1035.6         25.3%       较差 - 可能存在瓶颈
1024×2048           0.010       1654.5         40.4%       一般 - 有优化空间
1024×4096           0.015       2297.3         56.1%       良好 - 带宽利用合理
1024×8192           0.024       2794.6         68.2%       良好 - 带宽利用合理
1024×16384          0.062       2170.3         53.0%       良好 - 带宽利用合理
1024×32768          0.369       726.5          17.7%       较差 - 可能存在瓶颈

============================================================
```

性能呈现先升后降的趋势，在 `n_cols` 较小时，受限于 Launch Overhead 或并行度不足；在 8192 左右达到带宽利用率的峰值；当列数进一步增加（如 32768），由于寄存器压力过大导致 Spill 或 Occupancy 急剧下降，性能出现崩盘。

### 5.3 进阶优化：两阶段 Reduction

当 `n_cols` 超过单个 block 能处理的范围时，需要将每行分成多个 block：

**思路**：
1. 每行分成 `K = ceil(n_cols / BLOCK_SIZE)` 个 block
2. 每个 block 先内部 reduction，得到局部 `(max_k, sum_k)`
3. 第二个 kernel 合并所有 block 的结果

```python
@triton.jit
def softmax_merge_kernel(
    partial_max_ptr,  # [M, K] 每个 block 的局部最大值
    partial_sum_ptr,  # [M, K] 每个 block 的局部指数和
    output_ptr,
    n_rows, n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    k_offsets = tl.arange(0, BLOCK_SIZE)
    mask = k_offsets < n_blocks

    # 加载该行所有 block 的局部结果
    partial_max = tl.load(partial_max_ptr + row_idx * n_blocks + k_offsets, mask=mask)
    partial_sum = tl.load(partial_sum_ptr + row_idx * n_blocks + k_offsets, mask=mask)

    # 跨 block reduction
    ...

    # 计算最终结果（这里需要重新加载原始数据，或者保存中间结果）
    # ... 省略细节 ...
```

**性能权衡**：

- 优势：可以处理任意大的 `n_cols`
- 劣势：需要两次 kernel 启动，额外的全局内存访问

完整的两阶段 reduction 实现请参考 [`homework.ipynb`](./homework.ipynb)。

## 六、总结

完成本节后，你应该理解了 Triton 的 Reduction 操作（`tl.max`, `tl.sum`），掌握了数值稳定的 Softmax 实现技巧，也理解了 Mask 机制在边界处理中的应用。


## 七、课后练习

请打开 [homework.ipynb](./homework.ipynb) 完成以下练习：

实现支持超大 `n_cols` 的 Softmax（两阶段 Reduction）

当前实现的 Softmax 要求 `BLOCK_SIZE >= n_cols`，受限于 GPU 寄存器数量。当 `n_cols` 很大时（如 8192、16384），需要将每行分成多个 block 处理，这就是两阶段 Reduction 技术：

- Stage 1：每个 block 计算自己负责的列段的局部 (max, sum)
- Stage 2：合并所有 block 的结果，得到最终的 softmax

这个练习会让你深入理解 Triton 中如何处理跨 block 的数据归约，也是实际生产中处理大规模数据的常用技巧。

## 参考资料

1. https://triton-lang.org/main/getting-started/tutorials/05-softmax.html
2. https://arxiv.org/abs/1706.03762
3. https://arxiv.org/abs/2205.14135