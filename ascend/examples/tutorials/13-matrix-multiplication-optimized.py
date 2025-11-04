# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

from prof_util import profiler_wrapper, compare_profiling_results, print_profiling_summary


# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


# Kernel using diagonal core allocation strategy (optimized for large matrices)
@triton.jit
def matmul_kernel_diagonal(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    '''
    8 * 8 对角线分核方式中,每8 * 8分格内任务块编号如下
    [0,  8,  16, 24, 32, 40, 48, 56]
    [57, 1,  9,  17, 25, 33, 41, 49]
    [50, 58, 2,  10, 18, 26, 34, 42]
    [43, 51, 59, 3,  11, 19, 27, 35]
    [36, 44, 52, 60, 4,  12, 20, 28]
    [29, 37, 45, 53, 61, 5,  13, 21]
    [22, 30, 38, 46, 54, 62, 6,  14]
    [15, 23, 31, 39, 47, 55, 63, 7]

    对角线分核可以明显减小Bank冲突,提升L2Cache利用率
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Always use diagonal allocation strategy
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        #8 * 8 对角线分核代码实现
        curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
        localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
        task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
        #求最小公倍数，方便求基本块的坐标
        x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                k_start + tl.arange(0, BLOCK_K)
            )[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (k_start + tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)


# Kernel using swizzle2d for optimized memory access patterns
@triton.jit
def matmul_kernel_swizzle2d(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    '''
    使用 swizzle2d 优化内存访问模式
    swizzle2d 可以改善内存访问的局部性，减少 bank conflicts
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Use swizzle2d for block allocation
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            # Use swizzle2d to optimize memory access pattern for mat_a
            m_range = tl.arange(0, BLOCK_M)
            k_range = tl.arange(0, BLOCK_K)
            m_swizzled, k_swizzled = tl.swizzle2d(m_range, k_range,
                                                   size_i=BLOCK_M, size_j=BLOCK_K, size_g=16)

            mat_a_offset = ((m_start + m_swizzled) * K)[:, None] + (
                k_start + k_range
            )[None, :]
            mat_a_mask = ((m_start + m_range) < M)[:, None] & (
                (k_start + k_range) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")

            # Use swizzle2d to optimize memory access pattern for mat_b
            n_range = tl.arange(0, BLOCK_N)
            k_swizzled_b, n_swizzled = tl.swizzle2d(k_range, n_range,
                                                     size_i=BLOCK_K, size_j=BLOCK_N, size_g=16)

            mat_b_offset = ((k_start + k_range) * N)[:, None] + (
                n_start + n_swizzled
            )[None, :]
            mat_b_mask = ((k_start + k_range) < K)[:, None] & (
                (n_start + n_range) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")

            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)

        # Store result with swizzle2d
        m_range = tl.arange(0, BLOCK_M)
        n_range = tl.arange(0, BLOCK_N)
        m_swizzled_out, n_swizzled_out = tl.swizzle2d(m_range, n_range,
                                                       size_i=BLOCK_M, size_j=BLOCK_N, size_g=16)

        mat_c_offset = ((m_start + m_swizzled_out) * N)[:, None] + (
            n_start + n_swizzled_out
        )[None, :]
        mat_c_mask = ((m_start + m_range) < M)[:, None] & (
            (n_start + n_range) < N
        )[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)


# Kernel using sequential core allocation strategy (traditional approach)
@triton.jit
def matmul_kernel_sequential(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    '''
    水平分核方式每个任务块编号如下
    [0,  1,  2,  3,  4,  5,  6,  7]
    [8,  9,  10, 11, 12, 13, 14, 15]
    [16, 17, 18, 19, 20, 21, 22, 23]
    [24, 25, 26, 27, 28, 29, 30, 31]
    [32, 33, 34, 35, 36, 37, 38, 39]
    [40, 41, 42, 43, 44, 45, 46, 47]
    [48, 49, 50, 51, 52, 53, 54, 55]
    [56, 57, 58, 59, 60, 61, 62, 63]

    传统顺序分核策略
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Always use sequential allocation strategy
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                k_start + tl.arange(0, BLOCK_K)
            )[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (k_start + tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)

def triton_matmul(
    mat_a,
    mat_b,
    kernel_type='diagonal',  # 'diagonal', 'sequential', or 'swizzle2d'
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=256,
    BLOCK_TRESHHOLD=8,
):
    """
    Triton matrix multiplication with selectable core allocation strategy.

    Args:
        mat_a: Input matrix A
        mat_b: Input matrix B
        kernel_type: 'diagonal' for optimized diagonal allocation,
                    'sequential' for traditional approach,
                    'swizzle2d' for swizzle2d optimized memory access
        BLOCK_M, BLOCK_N, BLOCK_K: Block sizes for tiling
        BLOCK_TRESHHOLD: Threshold for diagonal allocation (only used for diagonal kernel)

    Returns:
        Output matrix C = A @ B
    """
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)

    num_cores = get_npu_properties()["num_aicore"]

    if kernel_type == 'diagonal':
        matmul_kernel_diagonal[(num_cores,)](
            mat_a, mat_b, mat_c,
            m, n, k, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD
        )
    elif kernel_type == 'sequential':
        matmul_kernel_sequential[(num_cores,)](
            mat_a, mat_b, mat_c,
            m, n, k, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
    elif kernel_type == 'swizzle2d':
        matmul_kernel_swizzle2d[(num_cores,)](
            mat_a, mat_b, mat_c,
            m, n, k, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Must be 'diagonal', 'sequential', or 'swizzle2d'")

    return mat_c


def run_benchmark(M, K, N, kernel_type, result_paths, BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, BLOCK_TRESHHOLD=8):
    """
    Run benchmark for a specific kernel type.

    Args:
        M, K, N: Matrix dimensions
        kernel_type: 'diagonal' or 'sequential'
        result_paths: Dictionary to store profiling result paths
        BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD: Kernel parameters
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {kernel_type} kernel with M={M}, K={K}, N={N}")
    print(f"Block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    if kernel_type == 'diagonal':
        print(f"BLOCK_TRESHHOLD={BLOCK_TRESHHOLD}")
    print(f"{'=' * 80}")

    # Create test matrices
    mat_a = torch.randn([M, K], dtype=torch.bfloat16, device="npu")
    mat_b = torch.randn([K, N], dtype=torch.bfloat16, device="npu")

    # Test correctness first
    result = triton_matmul(mat_a, mat_b, kernel_type=kernel_type,
                          BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                          BLOCK_TRESHHOLD=BLOCK_TRESHHOLD)
    golden = torch.matmul(mat_a, mat_b)

    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    try:
        torch.testing.assert_close(result[mask], golden[mask], atol=tmpatol, rtol=0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=tmprtol)
        print(f"✓ {kernel_type} kernel correctness check PASSED")
    except Exception as e:
        print(f"✗ {kernel_type} kernel correctness check FAILED")
        print(f"  Error: {e}")
        return

    # Profile performance
    def kernel_wrapper():
        triton_matmul(mat_a, mat_b, kernel_type=kernel_type,
                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                     BLOCK_TRESHHOLD=BLOCK_TRESHHOLD)

    result_path = f"./result_profiling_{kernel_type}"
    print(f"\nProfiling {kernel_type} kernel...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for later comparison
    result_paths[kernel_type] = result_path



if __name__ == "__main__":
    # Test configuration
    M = 2048
    K = 7168
    N = 16384

    # Block sizes (optimized for NPU - 512B alignment friendly)
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    BLOCK_TRESHHOLD = 8  # 8x8 diagonal allocation

    print("\n" + "=" * 80)
    print("Matrix Multiplication Performance Comparison")
    print("=" * 80)
    print(f"Matrix dimensions: A[{M}, {K}] @ B[{K}, {N}] = C[{M}, {N}]")
    print(f"Data type: bfloat16")
    print("=" * 80)

    # Dictionary to store profiling result paths
    profiling_results = {}

    # Run benchmarks for all three kernel types
    print("\nRunning benchmarks for three different optimization strategies:")
    print("  1. Sequential: Traditional row-major allocation")
    print("  2. Diagonal: Optimized diagonal allocation for better cache utilization")
    print("  3. Swizzle2D: Using swizzle2d for optimized memory access patterns")
    print("=" * 80)

    run_benchmark(M, K, N, 'sequential', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)
    run_benchmark(M, K, N, 'diagonal', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)
    run_benchmark(M, K, N, 'swizzle2d', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)

    # Compare and report profiling results
    print("\n" + "=" * 80)
    print("Performance Comparison: Three Optimization Strategies")
    print("=" * 80)

    results = compare_profiling_results(profiling_results)
    print_profiling_summary(results,
                          title="Matrix Multiplication: Sequential vs Diagonal vs Swizzle2D")

    print("\nOptimization Strategy Summary:")
    print("-" * 80)
    print("Sequential Allocation:")
    print("  - Traditional row-major block allocation")
    print("  - Simple and straightforward")
    print("  - May suffer from bank conflicts and cache misses on large matrices")
    print()
    print("Diagonal Allocation:")
    print("  - Optimized diagonal block allocation")
    print("  - Reduces bank conflicts by distributing memory access patterns")
    print("  - Improved L2 cache utilization for large right matrices")
    print("  - Better performance when NUM_BLOCKS_M and NUM_BLOCKS_N >= BLOCK_TRESHHOLD")
    print()
    print("Swizzle2D Allocation:")
    print("  - Uses tl.swizzle2d to optimize memory access patterns")
    print("  - Improves memory access locality")
    print("  - Reduces bank conflicts through access pattern transformation")
    print("  - Particularly effective for certain memory layouts")
    print("=" * 80 + "\n")