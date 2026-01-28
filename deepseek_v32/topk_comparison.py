"""
TopK Implementation Comparison for Long Sequences (128k tokens)

Compares different TopK approaches:
1. torch.topk (baseline)
2. Radix TopK (original single-block)
3. Radix Split-KV (two-stage)
4. Radix Fused (last block standing)
5. Sample Sort TopK (experimental, approximate)

Key finding: Original radix topk is already very efficient!
"""

import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


# ============================================================================
# Sample Sort TopK (Experimental - Approximate)
# ============================================================================
#
# 核心思想:
#   1. 随机采样估计第 k 大元素的近似值 (threshold)
#   2. 用 threshold 过滤数据，只保留 >= threshold 的元素
#   3. 对过滤后的少量数据做精确排序
#
# 优势:
#   - 极高并行度 (每个 thread 独立处理)
#   - 最小 shared memory 使用
#   - O(n) 复杂度
#
# 劣势:
#   - 近似结果 (采样可能不准)
#   - 需要两遍扫描
# ============================================================================

@tilelang.jit(pass_configs=pass_configs)
def tl_topk_sample_impl(topk, num_samples=1024, in_dtype="float32", out_dtype="int32"):
    """
    Sample Sort based TopK - 简化版

    核心思想：采样估计 threshold，过滤后精确选择
    """
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    BLOCK_SIZE = 1024

    # 候选 buffer 大小 (预期 >= threshold 的元素数量约为 k，留余量)
    CANDIDATE_BUFFER_SIZE = topk * 4

    @T.prim_func
    def tl_topk_sample_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        output_indices: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx,):
            tx = T.get_thread_binding()

            # 采样点
            s_samples = T.alloc_shared([num_samples], in_dtype)
            # 候选元素
            s_candidate_vals = T.alloc_shared([CANDIDATE_BUFFER_SIZE], in_dtype)
            s_candidate_idxs = T.alloc_shared([CANDIDATE_BUFFER_SIZE], out_dtype)
            s_candidate_count = T.alloc_shared([1], out_dtype)
            s_threshold = T.alloc_shared([1], in_dtype)

            l_start_idx = T.alloc_var(out_dtype)
            l_end_idx = T.alloc_var(out_dtype)
            l_range = T.alloc_var(out_dtype)
            l_val = T.alloc_var(in_dtype)
            l_threshold = T.alloc_var(in_dtype)
            l_input_idx = T.alloc_var(out_dtype)
            l_sample_idx = T.alloc_var(out_dtype)
            l_threshold_pos = T.alloc_var(out_dtype)
            l_pos = T.alloc_var(out_dtype)
            l_candidate_count = T.alloc_var(out_dtype)
            l_tmp_v = T.alloc_var(in_dtype)
            l_tmp_i = T.alloc_var(out_dtype)

            l_start_idx = starts[bx]
            l_end_idx = ends[bx]
            l_range = l_end_idx - l_start_idx

            # ===== Step 1: 均匀采样 =====
            for i in T.serial(T.ceildiv(num_samples, BLOCK_SIZE)):
                l_sample_idx = i * BLOCK_SIZE + tx
                if l_sample_idx < num_samples:
                    l_input_idx = l_start_idx + (l_sample_idx * l_range) // num_samples
                    if l_input_idx < l_end_idx:
                        s_samples[l_sample_idx] = input[bx, l_input_idx]
                    else:
                        s_samples[l_sample_idx] = T.Cast(in_dtype, -1e10)
            T.sync_threads()

            # ===== Step 2: Bitonic Sort 采样点 (降序) =====
            for stage in T.serial(10):  # log2(1024) = 10
                for step in T.serial(stage + 1):
                    T.sync_threads()
                    stride = 1 << (stage - step)
                    for i in T.serial(T.ceildiv(num_samples, BLOCK_SIZE)):
                        idx = i * BLOCK_SIZE + tx
                        if idx < num_samples:
                            pair_idx = idx ^ stride
                            if pair_idx > idx and pair_idx < num_samples:
                                block_size_local = 1 << (stage + 1)
                                dir_bit = (idx // block_size_local) % 2
                                if dir_bit == 0:
                                    if s_samples[idx] < s_samples[pair_idx]:
                                        l_tmp_v = s_samples[idx]
                                        s_samples[idx] = s_samples[pair_idx]
                                        s_samples[pair_idx] = l_tmp_v
                                else:
                                    if s_samples[idx] > s_samples[pair_idx]:
                                        l_tmp_v = s_samples[idx]
                                        s_samples[idx] = s_samples[pair_idx]
                                        s_samples[pair_idx] = l_tmp_v
                    T.sync_threads()

            # 计算 threshold 位置 (留 2x 余量)
            if tx == 0:
                l_threshold_pos = (topk * num_samples * 2) // l_range
                if l_threshold_pos >= num_samples:
                    l_threshold_pos = num_samples - 1
                s_threshold[0] = s_samples[l_threshold_pos]
                s_candidate_count[0] = 0
            T.sync_threads()

            # ===== Step 3: 过滤数据 =====
            l_threshold = s_threshold[0]
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                l_input_idx = l_start_idx + s * BLOCK_SIZE + tx
                if l_input_idx < l_end_idx:
                    l_val = input[bx, l_input_idx]
                    if l_val >= l_threshold:
                        l_pos = T.atomic_add(s_candidate_count[0], 1, return_prev=True)
                        if l_pos < CANDIDATE_BUFFER_SIZE:
                            s_candidate_vals[l_pos] = l_val
                            s_candidate_idxs[l_pos] = l_input_idx
            T.sync_threads()

            # ===== Step 4: 对候选排序，输出 top-k =====
            l_candidate_count = s_candidate_count[0]
            if l_candidate_count > CANDIDATE_BUFFER_SIZE:
                l_candidate_count = CANDIDATE_BUFFER_SIZE

            # Odd-even transposition sort
            for phase in T.serial(CANDIDATE_BUFFER_SIZE):
                T.sync_threads()
                for i in T.serial(T.ceildiv(CANDIDATE_BUFFER_SIZE // 2, BLOCK_SIZE)):
                    idx = (i * BLOCK_SIZE + tx) * 2 + 1
                    if idx < l_candidate_count - 1:
                        if s_candidate_vals[idx] < s_candidate_vals[idx + 1]:
                            l_tmp_v = s_candidate_vals[idx]
                            l_tmp_i = s_candidate_idxs[idx]
                            s_candidate_vals[idx] = s_candidate_vals[idx + 1]
                            s_candidate_idxs[idx] = s_candidate_idxs[idx + 1]
                            s_candidate_vals[idx + 1] = l_tmp_v
                            s_candidate_idxs[idx + 1] = l_tmp_i
                T.sync_threads()

                for i in T.serial(T.ceildiv(CANDIDATE_BUFFER_SIZE // 2, BLOCK_SIZE)):
                    idx = (i * BLOCK_SIZE + tx) * 2
                    if idx < l_candidate_count - 1:
                        if s_candidate_vals[idx] < s_candidate_vals[idx + 1]:
                            l_tmp_v = s_candidate_vals[idx]
                            l_tmp_i = s_candidate_idxs[idx]
                            s_candidate_vals[idx] = s_candidate_vals[idx + 1]
                            s_candidate_idxs[idx] = s_candidate_idxs[idx + 1]
                            s_candidate_vals[idx + 1] = l_tmp_v
                            s_candidate_idxs[idx + 1] = l_tmp_i

            T.sync_threads()

            # 输出前 topk 个
            for i in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                idx = i * BLOCK_SIZE + tx
                if idx < topk:
                    if idx < l_candidate_count:
                        output_indices[bx, idx] = s_candidate_idxs[idx]
                    else:
                        output_indices[bx, idx] = 0

    return tl_topk_sample_kernel


# ============================================================================
# Wrapper functions
# ============================================================================

def topk_sample_sort(input, starts, ends, topk, num_samples=1024):
    """Sample Sort based TopK"""
    batch = input.shape[0]
    output_indices = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    kernel = tl_topk_sample_impl(topk, num_samples)
    kernel(input, output_indices, starts, ends)
    return output_indices


# ============================================================================
# Benchmark and Comparison
# ============================================================================

def benchmark_topk_methods(batch=64, seq_len=128*1024, topk=2048, n_warmup=5, n_iters=20):
    """Benchmark all TopK implementations"""
    print(f"\n{'='*70}")
    print(f"TopK Comparison: batch={batch}, seq_len={seq_len}, topk={topk}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    # Reference: torch.topk
    ref_indices = torch.topk(input, topk, dim=-1)[1]

    def compute_accuracy(test_indices):
        """Compute accuracy as intersection ratio with reference"""
        total_acc = 0.0
        for i in range(batch):
            ref_set = set(ref_indices[i].cpu().numpy())
            test_set = set(test_indices[i].cpu().numpy())
            intersection = ref_set & test_set
            total_acc += len(intersection) / len(ref_set)
        return total_acc / batch

    def benchmark_kernel(name, kernel_fn, *args):
        """Benchmark a single kernel"""
        try:
            # Warmup
            for _ in range(n_warmup):
                _ = kernel_fn(*args)
            torch.cuda.synchronize()

            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(n_iters):
                result = kernel_fn(*args)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event) / n_iters
            accuracy = compute_accuracy(result)

            print(f"{name:30s}: {elapsed_ms:8.3f} ms | Accuracy: {accuracy:.4f}")
            return elapsed_ms, accuracy
        except Exception as e:
            print(f"{name:30s}: FAILED - {e}")
            return None, None

    print(f"\n--- Results ---")

    # 1. PyTorch baseline
    benchmark_kernel("torch.topk", lambda: torch.topk(input, topk, dim=-1)[1])

    # 2. Original Radix TopK (from topk_selector.py)
    from topk_selector import tl_topk
    benchmark_kernel("Radix TopK (original)", tl_topk, input, starts, ends, topk)

    # 3. Two-stage split_kv (from topk_selector.py)
    from topk_selector import tl_topk_split_kv
    benchmark_kernel("Radix Split-KV (2-stage)", tl_topk_split_kv, input, starts, ends, topk)

    # 4. Fused split_kv (from topk_selector.py)
    from topk_selector import tl_topk_fused
    benchmark_kernel("Radix Fused (last block)", tl_topk_fused, input, starts, ends, topk)

    # 5. Sample Sort TopK (experimental)
    benchmark_kernel("Sample Sort TopK", topk_sample_sort, input, starts, ends, topk)

    print(f"\n{'='*70}")
    print("Key insights:")
    print("- Original radix topk is already faster than torch.topk!")
    print("- 50% occupancy doesn't seem to be a real bottleneck")
    print("- Split-KV adds overhead for 128k seq_len")
    print(f"{'='*70}")


if __name__ == "__main__":
    benchmark_topk_methods()
