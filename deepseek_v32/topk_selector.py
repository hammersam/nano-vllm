import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def convert_to_uint16(x):
    hval = T.Cast("float16", x)
    bits_uint = T.reinterpret("uint16", hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret("uint32", x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast("uint32", (0xFFFFFFFF)),
        bits_uint | T.Cast("uint32", (0x80000000)),
    )
    return bits_uint


# ============================================================================
# Two-stage TopK for long context (e.g., 128k tokens)
# Stage 1: Multiple blocks process chunks in parallel, each selects local topk
# Stage 2: One block merges all chunk candidates to select final topk
# ============================================================================

# TODO: context length > 128k?
MAX_CONTEXT_LENGTH = 128 * 1024

# MAX_CHUNKS = 8
# CHUNK_SIZE = MAX_CONTEXT_LENGTH // MAX_CHUNKS

# 4:1性能最好，选择合适的seme_input_sz甚至能够跑过tl_topk_single_block
MAX_CHUNKS = 4
CHUNK_SIZE = MAX_CONTEXT_LENGTH // MAX_CHUNKS

# MAX_CHUNKS = 2
# CHUNK_SIZE = MAX_CONTEXT_LENGTH // MAX_CHUNKS

# Each chunk processes CHUNK_SIZE elements and selects topk candidates.
# threshold bin may hold up to topk elements in worst case.
# 这个threshold bin大小究竟应该设置成多少仍然未知，因为小于2348(e.g., 2048)会导致程序崩溃报错，
# 而大于这个值例如4096虽然更加安全，但是似乎比较浪费shared mem？
split_smem_input_sz = 2348
merge_smem_input_sz = 2348
fused_smem_input_sz = 2348
single_smem_input_sz = 2348

@tilelang.jit(pass_configs=pass_configs)
def tl_topk_stage1_impl(topk, chunk_size=CHUNK_SIZE, in_dtype="float32", out_dtype="int32", smem_input_size=4096):
    """Stage 1: Each block processes one chunk and selects local topk candidates."""
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    num_chunks = T.symbolic("num_chunks")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = smem_input_size

    @T.prim_func
    def tl_topk_stage1_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        chunk_indices: T.Tensor[(batch, num_chunks, topk), out_dtype],
        chunk_counts: T.Tensor[(batch, num_chunks), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, num_chunks, threads=BLOCK_SIZE) as (bx, by):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")
            s_output_count = T.alloc_shared([1], "int32")

            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_chunk_start = T.alloc_var("int32")
            l_chunk_end = T.alloc_var("int32")
            l_start_idx = T.alloc_var("int32")
            l_end_idx = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")
            l_chunk_len = T.alloc_var("int32")
            l_pos = T.alloc_var("int32")

            # Calculate this chunk's range
            l_chunk_start = by * chunk_size
            l_chunk_end = (by + 1) * chunk_size
            if l_chunk_end > seq_len:
                l_chunk_end = seq_len

            # Intersect with starts/ends
            l_start_idx = starts[bx]
            if l_start_idx < l_chunk_start:
                l_start_idx = l_chunk_start
            l_end_idx = ends[bx]
            if l_end_idx > l_chunk_end:
                l_end_idx = l_chunk_end

            l_chunk_len = l_end_idx - l_start_idx
            if l_chunk_len < 0:
                l_chunk_len = 0

            # Effective topk for this chunk (can't select more than chunk has)
            l_new_topk = topk
            if l_new_topk > l_chunk_len:
                l_new_topk = l_chunk_len

            # Initialize
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)
            if tx == 0:
                s_output_count[0] = 0
            T.sync_threads()

            # Skip if chunk is empty
            if l_chunk_len > 0:
                # Stage 1: 8-bit quick topk within this chunk
                for s in T.serial(T.ceildiv(chunk_size, BLOCK_SIZE)):
                    input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx:
                        inval_int16 = convert_to_uint16(input[bx, input_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                # Cumsum (suffix sum from high to low)
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # Find threshold bin id
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # Collect elements with bin_id > threshold
                for s in T.serial(T.ceildiv(chunk_size, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx:
                        bin_id = convert_to_uint16(input[bx, input_idx])
                        l_bin_id32 = T.Cast("int32", bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            chunk_indices[bx, by, l_pos] = input_idx
                            T.atomic_add(s_output_count[0], 1)
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, l_pos] = input_idx

                # Stage 2: tail pass for elements in threshold bin
                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()

                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((
                                convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                (24 - round * 8)) & 0xFF))
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()

                    # Cumsum
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[tx] = l_val

                        T.sync_threads(3, RADIX)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()

                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((
                                convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                (24 - round * 8)) & 0xFF))
                            if l_bin_id32 > l_threshold_bin_id:
                                l_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                chunk_indices[bx, by, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                T.atomic_add(s_output_count[0], 1)
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(
                                        s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        chunk_indices[bx, by, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                        T.atomic_add(s_output_count[0], 1)
                                else:
                                    l_pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

            # Write output count
            T.sync_threads()
            if tx == 0:
                chunk_counts[bx, by] = s_output_count[0]

    return tl_topk_stage1_kernel


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_stage2_impl(topk, max_chunks=MAX_CHUNKS, in_dtype="float32", out_dtype="int32", smem_input_size=4096):
    """Stage 2: Merge candidates from all chunks and select final topk."""
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    num_chunks = T.symbolic("num_chunks")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = smem_input_size

    @T.prim_func
    def tl_topk_stage2_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        chunk_indices: T.Tensor[(batch, num_chunks, topk), out_dtype],
        chunk_counts: T.Tensor[(batch, num_chunks), out_dtype],
        output_indices: T.Tensor[(batch, topk), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx,):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")
            s_total_candidates = T.alloc_shared([1], "int32")

            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")
            l_total = T.alloc_var("int32")
            l_chunk_count = T.alloc_var("int32")
            l_orig_idx = T.alloc_var("int32")
            l_pos = T.alloc_var("int32")

            # Count total candidates
            if tx == 0:
                s_total_candidates[0] = 0
                for c in T.serial(num_chunks):
                    s_total_candidates[0] = s_total_candidates[0] + chunk_counts[bx, c]
            T.sync_threads()
            l_total = s_total_candidates[0]

            l_new_topk = topk
            if l_new_topk > l_total:
                l_new_topk = l_total

            # Initialize
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)
            T.sync_threads()

            # Stage 1: 8-bit quick topk on merged candidates
            for c in T.serial(num_chunks):
                l_chunk_count = chunk_counts[bx, c]
                for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_chunk_count:
                        l_orig_idx = chunk_indices[bx, c, s * BLOCK_SIZE + tx]
                        inval_int16 = convert_to_uint16(input[bx, l_orig_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            # Cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Collect elements with bin_id > threshold
            for c in T.serial(num_chunks):
                l_chunk_count = chunk_counts[bx, c]
                for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_chunk_count:
                        l_orig_idx = chunk_indices[bx, c, s * BLOCK_SIZE + tx]
                        bin_id = convert_to_uint16(input[bx, l_orig_idx])
                        l_bin_id32 = T.Cast("int32", bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            output_indices[bx, l_pos] = l_orig_idx
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, l_pos] = l_orig_idx

            # Stage 2: tail pass
            for round in T.serial(4):
                if l_new_topk <= 0:
                    T.loop_break()

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast("int32", ((
                            convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                            (24 - round * 8)) & 0xFF))
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()

                # Cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast("int32", ((
                            convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                            (24 - round * 8)) & 0xFF))
                        if l_bin_id32 > l_threshold_bin_id:
                            l_pos = T.atomic_add(
                                s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            output_indices[bx, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                l_out_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    output_indices[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                l_pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_stage2_kernel


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_fused_impl(topk, chunk_size=CHUNK_SIZE, in_dtype="float32", out_dtype="int32", smem_input_size=4096):
    """Fused Stage 1 & 2: Uses atomic barrier to merge stages into a single kernel."""
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    num_chunks = T.symbolic("num_chunks")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = smem_input_size

    @T.prim_func
    def tl_topk_fused_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        output_indices: T.Tensor[(batch, topk), out_dtype],
        # Intermediate buffers (global memory)
        chunk_indices: T.Tensor[(batch, num_chunks, topk), out_dtype],
        chunk_counts: T.Tensor[(batch, num_chunks), out_dtype],
        barrier: T.Tensor[(batch), "int32"],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, num_chunks, threads=BLOCK_SIZE) as (bx, by):
            tx = T.get_thread_binding()

            # Shared memory (reused across stages)
            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")
            s_output_count = T.alloc_shared([1], "int32")
            s_is_last_block = T.alloc_shared([1], "int32")
            s_total_candidates = T.alloc_shared([1], "int32")

            # Variables
            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_chunk_start = T.alloc_var("int32")
            l_chunk_end = T.alloc_var("int32")
            l_start_idx = T.alloc_var("int32")
            l_end_idx = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")
            l_chunk_len = T.alloc_var("int32")
            l_pos = T.alloc_var("int32")
            l_chunk_count = T.alloc_var("int32")
            l_orig_idx = T.alloc_var("int32")

            # =========================================================
            # STAGE 1: Chunk TopK
            # =========================================================
            l_chunk_start = by * chunk_size
            l_chunk_end = T.min((by + 1) * chunk_size, seq_len)
            l_start_idx = T.max(starts[bx], l_chunk_start)
            l_end_idx = T.min(ends[bx], l_chunk_end)
            l_chunk_len = T.max(l_end_idx - l_start_idx, 0)
            l_new_topk = T.min(topk, l_chunk_len)

            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)
            if tx == 0:
                s_output_count[0] = 0
            T.sync_threads()

            if l_chunk_len > 0:
                # 1. Histogram
                for s in T.serial(T.ceildiv(chunk_size, BLOCK_SIZE)):
                    input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx:
                        inval_int16 = convert_to_uint16(input[bx, input_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                # 2. Find Threshold
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # 3. Collect Candidates
                for s in T.serial(T.ceildiv(chunk_size, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = l_chunk_start + s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx:
                        bin_id = convert_to_uint16(input[bx, input_idx])
                        l_bin_id32 = T.Cast("int32", bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            chunk_indices[bx, by, l_pos] = input_idx
                            T.atomic_add(s_output_count[0], 1)
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, l_pos] = input_idx

                # 4. Tail Pass
                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()
                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk
                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0: s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()
                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF))
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset: l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset: s_histogram[tx] = l_val
                        T.sync_threads(3, RADIX)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF))
                            if l_bin_id32 > l_threshold_bin_id:
                                l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                chunk_indices[bx, by, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                T.atomic_add(s_output_count[0], 1)
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        chunk_indices[bx, by, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                        T.atomic_add(s_output_count[0], 1)
                                else:
                                    l_pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

            T.sync_threads()
            if tx == 0:
                chunk_counts[bx, by] = s_output_count[0]

            # =========================================================
            # GLOBAL BARRIER (Last Block Standing)
            # =========================================================
            # Ensure writes to global memory are visible
            # TODO: 这里需要确保所有全局内存写入对于其他block可见，然后才能够做原子操作吗？
            
            if tx == 0:
                # Atomic increment barrier
                old_val = T.atomic_add(barrier[bx], 1, return_prev=True)
                s_is_last_block[0] = T.if_then_else(old_val == (num_chunks - 1), 1, 0)
            T.sync_threads()

            # =========================================================
            # STAGE 2: Merge (Only Last Block)
            # =========================================================
            if s_is_last_block[0] == 1:
                # Reset barrier for next run
                if tx == 0:
                    barrier[bx] = 0
                
                # Count total candidates
                if tx == 0:
                    s_total_candidates[0] = 0
                    for c in T.serial(num_chunks):
                        s_total_candidates[0] = s_total_candidates[0] + chunk_counts[bx, c]
                T.sync_threads()
                
                l_new_topk = T.min(topk, s_total_candidates[0])
                T.fill(s_histogram, 0)
                T.fill(s_num_input[0], 0)
                T.sync_threads()

                # Histogram (direct chunk access)
                for c in T.serial(num_chunks):
                    l_chunk_count = chunk_counts[bx, c]
                    for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_chunk_count:
                            l_orig_idx = chunk_indices[bx, c, s * BLOCK_SIZE + tx]
                            inval_int16 = convert_to_uint16(input[bx, l_orig_idx])
                            T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                # Cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # Collect elements with bin_id > threshold
                for c in T.serial(num_chunks):
                    l_chunk_count = chunk_counts[bx, c]
                    for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_chunk_count:
                            l_orig_idx = chunk_indices[bx, c, s * BLOCK_SIZE + tx]
                            bin_id = convert_to_uint16(input[bx, l_orig_idx])
                            l_bin_id32 = T.Cast("int32", bin_id)
                            if l_bin_id32 > l_threshold_bin_id:
                                l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                                output_indices[bx, l_pos] = l_orig_idx
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, l_pos] = l_orig_idx

                # Stage 2: tail pass
                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()

                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((
                                convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                (24 - round * 8)) & 0xFF))
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()

                    # Cumsum
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[tx] = l_val

                        T.sync_threads(3, RADIX)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()

                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast("int32", ((
                                convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                (24 - round * 8)) & 0xFF))
                            if l_bin_id32 > l_threshold_bin_id:
                                l_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                output_indices[bx, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(
                                        s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        output_indices[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                else:
                                    l_pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_fused_kernel


def tl_topk_split_kv(input, starts, ends, topk, profile=False):
    """Two-stage topk for long context scenarios."""
    batch, seq_len = input.shape
    num_chunks = min(MAX_CHUNKS, (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)

    if profile:
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()

    # Allocate intermediate buffers
    chunk_indices = torch.zeros(batch, num_chunks, topk, dtype=torch.int32, device=input.device)
    chunk_counts = torch.zeros(batch, num_chunks, dtype=torch.int32, device=input.device)
    output_indices = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)

    if profile:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"  Allocation: {(t1-t0)*1000:.3f} ms")

    # Stage 1: Each chunk selects local topk
    stage1_kernel = tl_topk_stage1_impl(topk, CHUNK_SIZE, smem_input_size=split_smem_input_sz)
    stage1_kernel(input, chunk_indices, chunk_counts, starts, ends)

    if profile:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        print(f"  Stage 1 (chunk topk): {(t2-t1)*1000:.3f} ms")

    # Stage 2: Merge and select final topk
    stage2_kernel = tl_topk_stage2_impl(topk, num_chunks, smem_input_size=merge_smem_input_sz)
    stage2_kernel(input, chunk_indices, chunk_counts, output_indices)

    if profile:
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f"  Stage 2 (merge): {(t3-t2)*1000:.3f} ms")
        print(f"  Total: {(t3-t0)*1000:.3f} ms")

    return output_indices


def tl_topk_fused(input, starts, ends, topk, profile=False):
    """Fused two-stage topk (Last Block Standing)."""
    batch, seq_len = input.shape
    num_chunks = min(MAX_CHUNKS, (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)
    
    # Allocate buffers
    chunk_indices = torch.zeros(batch, num_chunks, topk, dtype=torch.int32, device=input.device)
    chunk_counts = torch.zeros(batch, num_chunks, dtype=torch.int32, device=input.device)
    output_indices = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    barrier = torch.zeros(batch, dtype=torch.int32, device=input.device)

    if profile:
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()

    fused_kernel = tl_topk_fused_impl(topk, CHUNK_SIZE, smem_input_size=fused_smem_input_sz)
    # print(fused_kernel.get_kernel_source())
    fused_kernel(input, output_indices, chunk_indices, chunk_counts, barrier, starts, ends)

    if profile:
        torch.cuda.synchronize()
        print(f"  Fused Kernel: {(time.perf_counter()-t0)*1000:.3f} ms")

    return output_indices

@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(topk, in_dtype="float32", out_dtype="int32", smem_input_size=4096):
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = smem_input_size  # assume the threshold bucket size after first pass is less than 4K

    @T.prim_func
    def tl_topk_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")

            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_start_idx = T.alloc_var("int32")
            l_end_idx = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")
            l_pos = T.alloc_var("int32")

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # stage 1: use 8bit to do quick topk
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)

            T.sync_threads()
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            # cumsum
            # 对histogram进行后缀和
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                # 寻找阈值仓位(i.e., 刚好凑够topk数量元素的位置，需要对这个位置进行更细致地排列)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            # 还剩下多少待排序元素才能够凑够topk
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # collect all elements with exponent ≥ threshold
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.Cast("int32", bin_id)
                    if l_bin_id32 > l_threshold_bin_id:
                        # 计算这个元素在topk中对应的位置，需要知道在histogram中右边有多少个元素比它大
                        l_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, l_pos] = input_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # 在阈值仓位的元素需要进一步进行细分
                        l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        s_input_idx[0, l_pos] = input_idx

            # stage 2: tail pass
            for round in T.serial(4):
                if l_new_topk <= 0:
                    T.loop_break()

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        # 从最高位到最低位，每轮获取不同的8bit用于排序
                        l_bin_id32 = T.Cast("int32", ((
                            convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                            (24 - round * 8)) & 0xFF))
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                # 同样也需要加起来为histogram计算后缀和
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val
                    # find threshold bin id
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        # 同样也许记载阈值仓位，因为单纯比较最高8位可能不足以进行完全区分
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                # 减去这一轮完成区分的元素数量，这些元素都属于topk之中
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast("int32", ((
                            convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                            (24 - round * 8)) & 0xFF))
                        if l_bin_id32 > l_threshold_bin_id:
                            l_pos = T.atomic_add(
                                s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            # l_new_topk表示要从threshold bin这个位置获取元素的个数
                            if round == 3:
                                # 如果到了round3两个元素的bin_id32还相等，说明真的相等？
                                l_out_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                # 记载需要留到下一轮处理的元素数量和索引
                                # 注意使用了不同的buf来避免冲突
                                l_pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, l_pos] = s_input_idx[r_idx,
                                                                          s * BLOCK_SIZE + tx]

    return tl_topk_kernel


def tl_topk(input, starts, ends, topk):
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    kernel = tl_topk_impl(topk, smem_input_size=single_smem_input_sz)
    kernel(input, indexes, starts, ends)
    return indexes


def test_topk_selector(batch=64, seq_len=32 * 1024, topk=2048):

    batch = 64
    seq_len = 32 * 1024
    topk = 2048
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = tl_topk(input, starts, ends, topk)
    print(indexes)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    print(indexes_ref)

    # Calculate intersection of out_ref and out_trt
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=",
              len(intersection) / len(set_ref))

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    n_iters = 20
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average tl_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        _ = torch.topk(input, topk, dim=-1)[1]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


def test_topk_split_kv(batch=64, seq_len=128 * 1024, topk=2048):
    """Test two-stage topk for long context (128k tokens)."""
    print(f"\n{'='*60}")
    print(f"Testing two-stage TopK: batch={batch}, seq_len={seq_len}, topk={topk}")
    NUM_CHUNKS = min(MAX_CHUNKS, (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)
    print(f"num_chunks={NUM_CHUNKS}, chunk_size={CHUNK_SIZE // 1024}k, stage2 merge size={2048 * NUM_CHUNKS // 1024}k")
    print(f"split_smem_input_sz={split_smem_input_sz}, merge_smem_input_sz={merge_smem_input_sz}")
    print(f"single_smem_input_sz={single_smem_input_sz}, fused_smem_input_sz={fused_smem_input_sz}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    # Test correctness
    print("\n--- Correctness Test ---")
    indexes_split = tl_topk_split_kv(input, starts, ends, topk, profile=False)
    indexes_ref = torch.topk(input, topk, dim=-1)[1]

    # Calculate intersection
    total_accuracy = 0.0
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        split_np = indexes_split[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_split = set(split_np)
        intersection = set_ref & set_split
        accuracy = len(intersection) / len(set_ref)
        total_accuracy += accuracy
        if i < 10:  # Print first 10 batches
            print(f"Batch {i}: selected/all = {len(intersection)}/{len(set_ref)} = {accuracy:.4f}")

    avg_accuracy = total_accuracy / batch
    print(f"Average accuracy: {avg_accuracy:.4f}")

    # Profiling breakdown (with Python overhead)
    print("\n--- Profiling Breakdown (single call, includes Python overhead) ---")
    _ = tl_topk_split_kv(input, starts, ends, topk, profile=True)

    # CUDA events based profiling (pure kernel time)
    print("\n--- CUDA Events Profiling (pure kernel time, avg of 20 runs) ---")

    # Pre-allocate buffers to exclude allocation from timing
    num_chunks = min(MAX_CHUNKS, (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE)
    chunk_indices = torch.zeros(batch, num_chunks, topk, dtype=torch.int32, device=input.device)
    chunk_counts = torch.zeros(batch, num_chunks, dtype=torch.int32, device=input.device)
    output_indices = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)

    stage1_kernel = tl_topk_stage1_impl(topk, CHUNK_SIZE, smem_input_size=split_smem_input_sz)
    stage2_kernel = tl_topk_stage2_impl(topk, num_chunks, smem_input_size=merge_smem_input_sz)

    # Warmup
    for _ in range(5):
        stage1_kernel(input, chunk_indices, chunk_counts, starts, ends)
        stage2_kernel(input, chunk_indices, chunk_counts, output_indices)
    torch.cuda.synchronize()

    n_profile = 20
    evt_start = torch.cuda.Event(enable_timing=True)
    evt_mid = torch.cuda.Event(enable_timing=True)
    evt_end = torch.cuda.Event(enable_timing=True)

    stage1_times = []
    stage2_times = []

    for _ in range(n_profile):
        evt_start.record()
        stage1_kernel(input, chunk_indices, chunk_counts, starts, ends)
        evt_mid.record()
        stage2_kernel(input, chunk_indices, chunk_counts, output_indices)
        evt_end.record()
        torch.cuda.synchronize()
        stage1_times.append(evt_start.elapsed_time(evt_mid))
        stage2_times.append(evt_mid.elapsed_time(evt_end))

    avg_stage1 = sum(stage1_times) / len(stage1_times)
    avg_stage2 = sum(stage2_times) / len(stage2_times)
    print(f"  Stage 1 (chunk topk): {avg_stage1:.3f} ms  [grid: ({batch}, {num_chunks})]")
    print(f"  Stage 2 (merge):      {avg_stage2:.3f} ms  [grid: ({batch},)]")
    print(f"  Total kernel time:    {avg_stage1 + avg_stage2:.3f} ms")

    # Performance test
    print("\n--- Performance Test ---")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk_split_kv(input, starts, ends, topk)
    torch.cuda.synchronize()

    n_iters = 20

    # Two-stage split_kv time
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk_split_kv(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    split_time = start_event.elapsed_time(end_event) / n_iters
    print(f"tl_topk_split_kv (two-stage): {split_time:.3f} ms")

    # Warmup
    for _ in range(5):
        _ = tl_topk_fused(input, starts, ends, topk)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk_fused(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    fused_time = start_event.elapsed_time(end_event) / n_iters
    print(f"tl_topk_fused (fused):        {fused_time:.3f} ms")

    # Test correctness
    print("\n--- Correctness Test: fused vs ref ---")
    indexes_fused = tl_topk_fused(input, starts, ends, topk, profile=False)
    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    # Calculate intersection
    total_accuracy = 0.0
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        fused_np = indexes_fused[i].cpu().to(torch.int32).numpy()
        set_ref = set(ref_np)
        set_fused = set(fused_np)
        intersection = set_ref & set_fused
        accuracy = len(intersection) / len(set_ref)
        total_accuracy += accuracy
        if i < 10:  # Print first 10 batches
            print(f"Batch {i}: selected/all = {len(intersection)}/{len(set_ref)} = {accuracy:.4f}")

    # Single-block baseline time
    for _ in range(5):
        _ = tl_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    single_time = start_event.elapsed_time(end_event) / n_iters
    print(f"\n\ntl_topk (single-block):       {single_time:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        _ = torch.topk(input, topk, dim=-1)[1]
    end_event.record()
    torch.cuda.synchronize()
    torch_time = start_event.elapsed_time(end_event) / n_iters
    print(f"torch.topk:                   {torch_time:.3f} ms")

    print(f"\nSpeedup(split) vs single-block: {single_time / split_time:.2f}x")
    print(f"Speedup(split) vs torch.topk:   {torch_time / split_time:.2f}x")

    print(f"\nSpeedup(fused) vs single-block: {single_time / fused_time:.2f}x")
    print(f"Speedup(fused) vs torch.topk:   {torch_time / fused_time:.2f}x")

if __name__ == "__main__":
    # Test original single-block implementation
    # print("Testing single-block TopK (32k tokens):")
    # test_topk_selector()

    # Test two-stage implementation for long context
    # test_topk_split_kv(batch=64, seq_len=128 * 1024, topk=2048)

    print("\n--- small batch size ---")
    test_topk_split_kv(batch=4, seq_len=128 * 1024, topk=2048)
    print("\n--- large batch size ---")
    test_topk_split_kv(batch=64, seq_len=128 * 1024, topk=2048)

    # print("\n--- long context length ---")
    # MAX_CONTEXT_LENGTH = 8 * 128 * 1024
    # MAX_CHUNKS = 4 * 8
    # CHUNK_SIZE = MAX_CONTEXT_LENGTH // MAX_CHUNKS
    # split_smem_input_sz = 4096 * 2
    # merge_smem_input_sz = 4096 * 2
    # fused_smem_input_sz = 4096 * 2
    # single_smem_input_sz = 4096 * 2
    # test_topk_split_kv(batch=4, seq_len=8 * 128 * 1024, topk=2048)
