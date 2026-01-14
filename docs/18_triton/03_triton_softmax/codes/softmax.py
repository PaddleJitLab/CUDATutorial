import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    行级 Softmax Kernel：对输入矩阵的每一行独立做 softmax
    """
    # 每个 program 处理一行
    row_idx = tl.program_id(axis=0)

    # 计算该行的列偏移量（向量化）
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 计算行首地址
    row_start = x_ptr + row_idx * n_cols

    # 创建 mask：处理列数不是 BLOCK_SIZE 倍数的情况
    mask = col_offsets < n_cols

    # 加载一行数据
    # other=-float('inf')：mask=False 的位置用 -inf 填充
    x = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))

    # === 数值稳定的 Softmax ===
    # 1. 求行内最大值（用于数值稳定性）
    x_max = tl.max(x, axis=0)

    # 2. 减去最大值后计算指数
    x_exp = tl.exp(x - x_max)

    # 3. 求指数和
    x_sum = tl.sum(x_exp, axis=0)

    # 4. 归一化
    output = x_exp / x_sum

    # 写回结果
    out_row_start = output_ptr + row_idx * n_cols
    tl.store(out_row_start + col_offsets, output, mask=mask)


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


def test_softmax():
    """测试不同形状的输入"""
    test_cases = [
        (1024, 128),   # 常见尺寸
        (256, 64),     # 小尺寸
        (512, 256),    # 稍大尺寸
    ]

    for rows, cols in test_cases:
        print(f"Testing shape [{rows}, {cols}]...")

        # 随机输入
        x = torch.randn(rows, cols, device='cuda')

        # Triton vs PyTorch
        y_triton = softmax(x)
        y_torch = torch.nn.functional.softmax(x, dim=-1)

        # 检查误差
        max_error = torch.max(torch.abs(y_triton - y_torch)).item()
        print(f"  Max error: {max_error:.2e}")

        # 检查每行和为 1
        row_sums = y_triton.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

        assert torch.allclose(y_triton, y_torch, atol=1e-4)
        print("  ✓ Passed\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_softmax()
    print("All tests passed!")
