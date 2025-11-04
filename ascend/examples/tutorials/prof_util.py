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

"""
Profiler Utility
=================
Provides profiling utilities for Triton kernels on Ascend NPU.
"""

import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch_npu


def profiler_wrapper(fn, *args, result_path="./result_profiling", skip_first=10,
                     wait=0, warmup=3, active=30, repeat=1):
    """
    Wrapper function for profiling kernels using torch_npu profiler.

    Args:
        fn: Function to profile
        *args: Arguments to pass to the function
        result_path: Path to save profiling results (default: "./result_profiling")
        skip_first: Number of iterations to skip before profiling (default: 10)
        wait: Number of wait iterations in profiler schedule (default: 0)
        warmup: Number of warmup iterations in profiler schedule (default: 3)
        active: Number of active profiling iterations (default: 30)
        repeat: Number of times to repeat the profiling cycle (default: 1)
    """
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat,
                                                 skip_first=skip_first),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config) as prof:
        stream.synchronize()
        for i in range(skip_first + (wait + warmup + active) * repeat):
            fn(*args)
            prof.step()
        stream.synchronize()

    print(f"[INFO] Profiling results saved to {result_path}")
    print(f"[INFO] Check the kernel_details.csv file for detailed performance metrics")


def _find_latest_csv(result_path: str) -> Optional[Path]:
    """
    Find the latest kernel_details.csv file in the profiling output directory.
    The path structure is: result_path/xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv

    Args:
        result_path: Base profiling result path

    Returns:
        Path to the latest kernel_details.csv file, or None if not found
    """
    result_dir = Path(result_path)

    if not result_dir.exists():
        return None

    # Find all *_ascend_pt directories
    ascend_dirs = list(result_dir.glob("*_ascend_pt"))

    if not ascend_dirs:
        return None

    # Sort by modification time (newest first)
    ascend_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Find kernel_details.csv in the newest directory
    for ascend_dir in ascend_dirs:
        csv_path = ascend_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        if csv_path.exists():
            return csv_path

    return None


def parse_profiling_results(result_path: str) -> Optional[Dict]:
    """
    Parse profiling results from the kernel_details.csv file.
    Automatically detects column names from the CSV header.
    Handles the nested directory structure: result_path/xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/

    Args:
        result_path: Path to the profiling results directory

    Returns:
        Dictionary containing parsed profiling metrics, or None if file not found
    """
    # Find the actual CSV file in the nested directory structure
    csv_path = _find_latest_csv(result_path)

    if not csv_path:
        print(f"[WARNING] kernel_details.csv not found in {result_path}")
        print(f"[INFO] Expected structure: {result_path}/xxx_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        return None

    print(f"[INFO] Found CSV at: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                print(f"[WARNING] No data found in {csv_path}")
                return None

            # Print available columns for debugging (only first time)
            columns = list(rows[0].keys())
            print(f"[DEBUG] CSV columns: {columns}")

            # Find duration column - check common variations
            duration_col = None
            for col in columns:
                col_lower = col.lower()
                if 'duration' in col_lower and 'us' in col_lower:
                    duration_col = col
                    break
                elif col_lower in ['duration', 'dur', 'time', 'elapsed']:
                    duration_col = col
                    break
                elif 'task duration' in col_lower:
                    duration_col = col
                    break

            # Find name column
            name_col = None
            for col in columns:
                col_lower = col.lower()
                if col_lower in ['name', 'kernel name', 'op name', 'operator name']:
                    name_col = col
                    break

            if not duration_col:
                print(f"[ERROR] Could not find duration column. Available columns: {columns}")
                return None

            print(f"[INFO] Using duration column: '{duration_col}'")
            if name_col:
                print(f"[INFO] Using name column: '{name_col}'")

            # Aggregate metrics across all kernel invocations
            total_duration = 0.0
            call_count = 0
            kernel_names = set()

            for row in rows:
                try:
                    # Parse duration
                    if duration_col and row[duration_col]:
                        duration = float(row[duration_col])
                        total_duration += duration
                        call_count += 1

                    # Collect kernel names
                    if name_col and row[name_col]:
                        kernel_names.add(row[name_col])
                except (ValueError, KeyError) as e:
                    continue  # Skip rows with invalid data

            if call_count == 0:
                print(f"[WARNING] No valid duration data found in {csv_path}")
                return None

            avg_duration = total_duration / call_count

            return {
                'result_path': result_path,
                'total_duration_us': total_duration,
                'avg_duration_us': avg_duration,
                'call_count': call_count,
                'kernel_names': list(kernel_names) if kernel_names else ['N/A'],
                'duration_column': duration_col
            }

    except Exception as e:
        print(f"[ERROR] Failed to parse {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_profiling_results(result_paths: Dict[str, str]) -> Dict[str, Dict]:
    """
    Compare profiling results from multiple runs.

    Args:
        result_paths: Dictionary mapping names to result paths
                     e.g., {'int32': './result_profiling_int32', 'int64': './result_profiling_int64'}

    Returns:
        Dictionary mapping names to their parsed metrics
    """
    results = {}

    for name, path in result_paths.items():
        metrics = parse_profiling_results(path)
        if metrics:
            results[name] = metrics

    return results


def print_profiling_summary(results: Dict[str, Dict], title: str = "Profiling Summary"):
    """
    Print a formatted summary of profiling results.

    Args:
        results: Dictionary of parsed profiling results from compare_profiling_results
        title: Title for the summary report
    """
    if not results:
        print("[WARNING] No profiling results to display")
        return

    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    # Print header
    print(f"\n{'Data Type':<15} {'Avg Time (us)':<20} {'Calls':<10}")
    print("-" * 80)

    # Sort by average duration for easier comparison
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_duration_us'])

    # Print each result
    for name, metrics in sorted_results:
        avg_us = metrics['avg_duration_us']
        calls = metrics['call_count']
        print(f"{name:<15} {avg_us:<20.6f} {calls:<10}")

    # Print speedup comparison (relative to slowest)
    print("\n" + "-" * 80)
    print("Speedup Comparison (relative to slowest):")
    print("-" * 80)

    if len(sorted_results) > 1:
        slowest_name, slowest_metrics = sorted_results[-1]
        slowest_time = slowest_metrics['avg_duration_us']

        print(f"{'Data Type':<15} {'Speedup':<20} {'vs Baseline':<30}")
        print("-" * 80)

        for name, metrics in sorted_results:
            speedup = slowest_time / metrics['avg_duration_us']
            baseline_info = f"({slowest_name})" if name != slowest_name else "(baseline)"
            print(f"{name:<15} {speedup:<20.2f}x {baseline_info:<30}")

    print("=" * 80 + "\n")
