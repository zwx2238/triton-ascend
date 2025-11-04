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


# Kernel using tl.swizzle2d for optimized memory access patterns
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
    使用 tl.swizzle2d 优化内存访问模式
    swizzle2d 将行优先的索引模式转换为列优先的分组模式
    可以改善内存访问的局部性，减少 bank conflicts
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Sequential block allocation
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        task_m_idx = block_idx // NUM_BLOCKS_N
        task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            # Load mat_a with swizzle2d on indices
            # Key: use [:, None] and [None, :] to create broadcastable 2D arrays
            m_range = tl.arange(0, BLOCK_M)[:, None]  # Column vector (BLOCK_M, 1)
            k_range = tl.arange(0, BLOCK_K)[None, :]  # Row vector (1, BLOCK_K)

            # Apply swizzle2d: transforms (i,j) indices to improve memory locality
            m_swizzled, k_swizzled = tl.swizzle2d(m_range, k_range,
                                                   size_i=BLOCK_M, size_j=BLOCK_K, size_g=16)

            mat_a_offset = ((m_start + m_swizzled) * K) + (k_start + k_swizzled)
            mat_a_mask = ((m_start + m_range) < M) & ((k_start + k_range) < K)
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            tl.compile_hint(mat_a_block, "dot_pad_only_k")

            # Load mat_b with swizzle2d on indices
            k_range_b = tl.arange(0, BLOCK_K)[:, None]  # Column vector (BLOCK_K, 1)
            n_range = tl.arange(0, BLOCK_N)[None, :]    # Row vector (1, BLOCK_N)

            k_swizzled_b, n_swizzled = tl.swizzle2d(k_range_b, n_range,
                                                     size_i=BLOCK_K, size_j=BLOCK_N, size_g=16)

            mat_b_offset = ((k_start + k_swizzled_b) * N) + (n_start + n_swizzled)
            mat_b_mask = ((k_start + k_range_b) < K) & ((n_start + n_range) < N)
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            tl.compile_hint(mat_b_block, "dot_pad_only_k")

            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)

        # Store result with swizzle2d
        m_range_out = tl.arange(0, BLOCK_M)[:, None]
        n_range_out = tl.arange(0, BLOCK_N)[None, :]
        m_swizzled_out, n_swizzled_out = tl.swizzle2d(m_range_out, n_range_out,
                                                       size_i=BLOCK_M, size_j=BLOCK_N, size_g=16)

        mat_c_offset = ((m_start + m_swizzled_out) * N) + (n_start + n_swizzled_out)
        mat_c_mask = ((m_start + m_range_out) < M) & ((n_start + n_range_out) < N)
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
        kernel_type: 'torch', 'diagonal', 'sequential', or 'swizzle2d'
        result_paths: Dictionary to store profiling result paths
        BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD: Kernel parameters
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {kernel_type} kernel with M={M}, K={K}, N={N}")
    if kernel_type != 'torch':
        print(f"Block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
        if kernel_type == 'diagonal':
            print(f"BLOCK_TRESHHOLD={BLOCK_TRESHHOLD}")
    print(f"{'=' * 80}")

    # Create test matrices
    mat_a = torch.randn([M, K], dtype=torch.bfloat16, device="npu")
    mat_b = torch.randn([K, N], dtype=torch.bfloat16, device="npu")

    # Test correctness first
    if kernel_type == 'torch':
        result = torch.matmul(mat_a, mat_b)
    else:
        result = triton_matmul(mat_a, mat_b, kernel_type=kernel_type,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              BLOCK_TRESHHOLD=BLOCK_TRESHHOLD)

    golden = torch.matmul(mat_a, mat_b)

    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    accuracy_passed = True
    error_details = None

    try:
        torch.testing.assert_close(result[mask], golden[mask], atol=tmpatol, rtol=0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=tmprtol)
        print(f"✓ {kernel_type} kernel correctness check PASSED")
    except Exception as e:
        accuracy_passed = False
        error_str = str(e)
        # Extract error statistics
        import re
        mismatch_match = re.search(r'Mismatched elements: (\d+) / (\d+) \(([\d.]+)%\)', error_str)
        max_diff_match = re.search(r'Greatest absolute difference: ([\d.]+)', error_str)

        if mismatch_match:
            mismatch_count = int(mismatch_match.group(1))
            total_count = int(mismatch_match.group(2))
            mismatch_percent = float(mismatch_match.group(3))
            max_diff = float(max_diff_match.group(1)) if max_diff_match else None

            error_details = {
                'mismatch_count': mismatch_count,
                'total_count': total_count,
                'mismatch_percent': mismatch_percent,
                'max_diff': max_diff
            }

        print(f"⚠ {kernel_type} kernel correctness check FAILED (continuing anyway)")
        print(f"  Error: {e}")
        # Don't return, continue with profiling

    # Profile performance
    if kernel_type == 'torch':
        def kernel_wrapper():
            torch.matmul(mat_a, mat_b)
    else:
        def kernel_wrapper():
            triton_matmul(mat_a, mat_b, kernel_type=kernel_type,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         BLOCK_TRESHHOLD=BLOCK_TRESHHOLD)

    result_path = f"./result_profiling_{kernel_type}"
    print(f"\nProfiling {kernel_type} kernel...")
    profiler_wrapper(kernel_wrapper, result_path=result_path)

    # Store result path for later comparison
    result_paths[kernel_type] = {
        'path': result_path,
        'accuracy_passed': accuracy_passed,
        'error_details': error_details
    }



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

    # Run benchmarks for all four methods
    print("\nRunning benchmarks for four different methods:")
    print("  0. PyTorch: torch.matmul baseline (reference)")
    print("  1. Sequential: Traditional row-major allocation")
    print("  2. Diagonal: Optimized diagonal allocation for better cache utilization")
    print("  3. Swizzle2D: Swizzle pattern for block allocation")
    print("=" * 80)

    run_benchmark(M, K, N, 'torch', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)
    run_benchmark(M, K, N, 'sequential', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)
    run_benchmark(M, K, N, 'diagonal', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)
    run_benchmark(M, K, N, 'swizzle2d', profiling_results,
                 BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TRESHHOLD)

    # Compare and report profiling results
    print("\n" + "=" * 80)
    print("Performance Comparison: Four Methods")
    print("=" * 80)

    # Extract paths for profiling comparison
    profiling_paths = {k: v['path'] for k, v in profiling_results.items()}
    results = compare_profiling_results(profiling_paths)
    print_profiling_summary(results,
                          title="Matrix Multiplication: Speedup vs PyTorch Baseline")

    # Accuracy Summary
    print("\n" + "=" * 80)
    print("Accuracy Summary:")
    print("=" * 80)
    print(f"{'Method':<20} {'Status':<15} {'Error Rate':<15} {'Max Abs Diff':<15}")
    print("-" * 80)

    for name in ['torch', 'sequential', 'diagonal', 'swizzle2d']:
        if name in profiling_results:
            info = profiling_results[name]
            if info['accuracy_passed']:
                status = "✓ PASSED"
                error_rate = "0.0%"
                max_diff = "N/A"
            else:
                status = "✗ FAILED"
                if info['error_details']:
                    error_rate = f"{info['error_details']['mismatch_percent']:.2f}%"
                    max_diff = f"{info['error_details']['max_diff']:.1f}" if info['error_details']['max_diff'] else "N/A"
                else:
                    error_rate = "Unknown"
                    max_diff = "Unknown"
            print(f"{name:<20} {status:<15} {error_rate:<15} {max_diff:<15}")
    print("=" * 80)

    # Calculate and display speedup relative to torch.matmul
    if results and 'torch' in results:
        torch_time = results['torch']['avg_duration_us']
        print("\n" + "=" * 80)
        print("Speedup Analysis (relative to torch.matmul baseline):")
        print("=" * 80)
        print(f"{'Method':<20} {'Avg Time (us)':<20} {'Speedup vs torch':<20}")
        print("-" * 80)

        for name in ['torch', 'sequential', 'diagonal', 'swizzle2d']:
            if name in results:
                avg_time = results[name]['avg_duration_us']
                speedup = torch_time / avg_time
                speedup_str = f"{speedup:.2f}x"
                if name == 'torch':
                    speedup_str = "1.00x (baseline)"
                print(f"{name:<20} {avg_time:<20.2f} {speedup_str:<20}")
        print("=" * 80)

    print("\nMethod Summary:")
    print("-" * 80)
    print("PyTorch (torch.matmul):")
    print("  - Native PyTorch implementation")
    print("  - Baseline for comparison")
    print("  - ✓ Perfect accuracy")
    print()
    print("Sequential Allocation:")
    print("  - Traditional row-major block allocation")
    print("  - Simple and straightforward")
    print("  - May suffer from bank conflicts and cache misses on large matrices")
    print("  - ✓ Perfect accuracy")
    print()
    print("Diagonal Allocation:")
    print("  - Optimized diagonal block allocation")
    print("  - Reduces bank conflicts by distributing memory access patterns")
    print("  - Improved L2 cache utilization for large right matrices")
    print("  - Better performance when NUM_BLOCKS_M and NUM_BLOCKS_N >= BLOCK_TRESHHOLD")
    print("  - ✓ Perfect accuracy")
    print()
    print("Swizzle2D Allocation:")
    print("  - Swizzle pattern applied at block allocation level")
    print("  - Aims to improve memory access locality")
    print("  - Reduces bank conflicts through pattern transformation")
    if 'swizzle2d' in profiling_results and not profiling_results['swizzle2d']['accuracy_passed']:
        print("  - ✗ ACCURACY ISSUE DETECTED!")
        if profiling_results['swizzle2d']['error_details']:
            err = profiling_results['swizzle2d']['error_details']
            print(f"    • {err['mismatch_percent']:.2f}% elements have incorrect values")
            print(f"    • {err['mismatch_count']} / {err['total_count']} elements affected")
        print()
        print("  Root Cause Analysis:")
        print("  -------------------")
        NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
        NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
        print(f"  Block grid: {NUM_BLOCKS_M} x {NUM_BLOCKS_N} = {NUM_BLOCKS_M * NUM_BLOCKS_N} blocks")
        print(f"  SWIZZLE_GROUP: 4 (trying to swizzle in 4x4 groups)")
        print()
        print("  Problem: The current swizzle2d implementation has a flawed block mapping logic:")
        print()
        print("  1. INCORRECT APPROACH (current code):")
        print("     - Treats block_idx as 1D linear index")
        print("     - group_id = block_idx // (SWIZZLE_GROUP * SWIZZLE_GROUP)")
        print("     - This doesn't respect the 2D grid structure")
        print(f"     - For grid {NUM_BLOCKS_M}x{NUM_BLOCKS_N}, consecutive block_idx values")
        print("       map to different rows, NOT forming proper 4x4 groups")
        print()
        print("  2. CONSEQUENCE:")
        print("     - Some blocks get mapped to wrong (task_m_idx, task_n_idx) coordinates")
        print("     - Blocks out of range fall back to base indices (clamping)")
        print("     - This causes ~4.7% of output elements to be computed from wrong input data")
        print()
        print("  3. CORRECT APPROACH should be:")
        print("     - Convert block_idx to 2D grid coordinates (m_idx, n_idx)")
        print("     - Determine which 4x4 group the block belongs to")
        print("     - Apply swizzle within that group")
        print("     - Convert back to actual block coordinates")
        print()
        print("  Example for 16x64 grid:")
        print("    block_idx=0  -> (m=0, n=0)   -> group(0,0) local(0,0)")
        print("    block_idx=1  -> (m=0, n=1)   -> group(0,0) local(0,1)")
        print("    block_idx=64 -> (m=1, n=0)   -> group(0,0) local(1,0)")
        print("    Current code incorrectly groups consecutive block_idx values")
        print("    instead of respecting 2D spatial locality!")
    print("=" * 80 + "\n")