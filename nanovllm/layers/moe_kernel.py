import torch
import triton
import triton.language as tl


@triton.jit
def _token_permutation_fwd_kernel(
    # Pointers to tensors
    src_ptr,
    dest_ptr,
    expert_indices_ptr,
    expert_offsets_ptr,
    expert_counters_ptr,
    # Tensor dimensions
    num_tokens,
    hidden_size,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Permutes tokens based on their expert assignment.
    Each token is moved to a new contiguous block of memory corresponding to its expert.
    This version uses atomic operations to determine the destination index for each token.
    """
    # Each program instance handles one token.
    pid = tl.program_id(axis=0)
    if pid >= num_tokens:
        return

    # Get the expert this token is assigned to.
    # FIXME: 这里应该获取num_experts_per_tok个expert？
    # expert_id = tl.load(expert_indices_ptr + pid * num_experts_per_tok + tl.range(0, num_experts_per_tok))
    expert_id = tl.load(expert_indices_ptr + pid)

    # Atomically increment the counter for this expert to get the token's
    # local index within its expert group.
    local_idx = tl.atomic_add(expert_counters_ptr + expert_id, 1)

    # Get the base offset for this expert's token block.
    base_offset = tl.load(expert_offsets_ptr + expert_id)
    
    # Calculate the destination token index in the permuted tensor.
    dest_token_idx = base_offset + local_idx

    # Compute source and destination pointers for the token's data.
    src_token_offset = pid * hidden_size
    dest_token_offset = dest_token_idx * hidden_size

    # Copy the token's hidden state from source to destination.
    # This is done in blocks for efficiency.
    for offset in range(0, hidden_size, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < hidden_size
        
        # Load a block of data from the source tensor.
        data_block = tl.load(src_ptr + src_token_offset + block_offsets, mask=mask)
        
        # Store the block of data in the destination tensor.
        tl.store(dest_ptr + dest_token_offset + block_offsets, data_block, mask=mask)


@triton.jit
def _token_permutation_bwd_kernel(
    src_ptr,
    dest_ptr,
    expert_indices_ptr,
    expert_offsets_ptr,
    expert_counters_ptr,
    num_tokens,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs the inverse permutation to scatter tokens back to their original positions.
    """
    pid = tl.program_id(axis=0)
    if pid >= num_tokens:
        return

    expert_id = tl.load(expert_indices_ptr + pid)
    local_idx = tl.atomic_add(expert_counters_ptr + expert_id, 1)
    base_offset = tl.load(expert_offsets_ptr + expert_id)
    src_token_idx = base_offset + local_idx

    src_token_offset = src_token_idx * hidden_size
    # FIXME: 这里好像也没有考虑一个token会被路由到`num_experts_per_tok`个expert
    # dest_token_offset = (pid * num_experts_per_tok + tl.range(0, num_experts_per_tok)) * hidden_size
    dest_token_offset = pid * hidden_size

    for offset in range(0, hidden_size, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < hidden_size
        data_block = tl.load(src_ptr + src_token_offset + block_offsets, mask=mask)
        tl.store(dest_ptr + dest_token_offset + block_offsets, data_block, mask=mask)


def token_permutation(x: torch.Tensor, expert_indices: torch.Tensor, num_experts: int, is_forward: bool):
    num_tokens, hidden_size = x.shape
    # expert_indices: (num_tokens * num_experts_per_tok,)
    # expert_counts: (num_experts,)
    # 记录发送给每个expert的token数量，其中一些expert可能接受到的token数量为0
    expert_counts = torch.bincount(expert_indices, minlength=num_experts)
    # expert_offsets: (num_experts,)
    # 属于每个expert的各个部分的token的起始位置
    # | tokens for expert0 | ... | tokens for expertN |
    # ^                    ^     ^
    expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32) - expert_counts
    
    # 最终结果应该按照expert id排序
    dest = torch.empty_like(x)
    expert_counters = torch.zeros(num_experts, dtype=torch.int32, device=x.device)

    grid = (num_tokens,)
    
    kernel = _token_permutation_fwd_kernel if is_forward else _token_permutation_bwd_kernel
    
    # Heuristic for block size
    BLOCK_SIZE = 128 if hidden_size > 2048 else 64

    kernel[grid](
        x,
        dest,
        expert_indices,
        expert_offsets,
        expert_counters,
        num_tokens,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dest, expert_counts


# 这是group gemm不是segmented gemm
@triton.jit
def segmented_gemm(
    a_ptr, b_ptr, c_ptr,
    expert_counts_ptr, expert_offsets_ptr, 
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_experts: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def invoke_segmented_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, expert_counts: torch.Tensor):
    """
    High-level wrapper for the segmented GEMM kernel.
    """
    assert len(a.shape) == 2
    assert len(b.shape) == 3
    assert len(c.shape) == 2
    assert a.shape[0] == c.shape[0]
    assert b.shape[2] == c.shape[1]
    assert a.shape[1] == b.shape[1]

    # M: num_tokens, D: hidden_size
    # N: intermediate_size(up_proj) or hidden_size(down_proj)
    M, D = a.shape
    num_experts, _, N = b.shape

    # This is a simplified grid calculation. A more robust implementation
    # would need to handle expert scheduling and memory management for results.
    grid = lambda META: (num_experts,)

    # This is a placeholder for the actual kernel launch.
    # A real implementation needs to iterate through experts and launch kernels.
    # For simplicity, we are launching one large kernel, which is not correct
    # for segmented GEMM but illustrates the structure.
    segmented_gemm[grid](
        a, b, c,
        expert_counts, None, # expert_offsets is not used in this simplified version
        M, N, D,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        num_experts=num_experts,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    return c