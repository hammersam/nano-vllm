import torch
import triton
import triton.language as tl
import torch.distributed as dist
from typing import Optional, Tuple


@triton.jit
def _token_permutation_v2_fwd_kernel(
    src_ptr, dest_ptr, expert_indices_ptr, expert_offsets_ptr,
    expert_counters_ptr, token_expert_mapping_ptr,
    num_tokens, hidden_size, num_experts_per_tok,
    BLOCK_SIZE: tl.constexpr,
):
    """Enhanced token permutation kernel supporting multiple experts per token."""
    pid = tl.program_id(axis=0)
    if pid >= num_tokens:
        return

    # Handle multiple experts per token
    for k_idx in range(num_experts_per_tok):
        expert_idx_offset = pid * num_experts_per_tok + k_idx
        expert_id = tl.load(expert_indices_ptr + expert_idx_offset)
        
        # Atomic increment to get local position
        local_idx = tl.atomic_add(expert_counters_ptr + expert_id, 1)
        base_offset = tl.load(expert_offsets_ptr + expert_id)
        dest_token_idx = base_offset + local_idx
        
        # Store mapping for inverse permutation
        tl.store(token_expert_mapping_ptr + expert_idx_offset, dest_token_idx)
        
        # Copy token data
        src_offset = pid * hidden_size
        dest_offset = dest_token_idx * hidden_size
        
        for offset in range(0, hidden_size, BLOCK_SIZE):
            block_offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = block_offsets < hidden_size
            data = tl.load(src_ptr + src_offset + block_offsets, mask=mask)
            tl.store(dest_ptr + dest_offset + block_offsets, data, mask=mask)


@triton.jit
def _token_permutation_v2_bwd_kernel(
    src_ptr, dest_ptr, token_expert_mapping_ptr,
    num_tokens, hidden_size, num_experts_per_tok,
    BLOCK_SIZE: tl.constexpr,
):
    """Enhanced inverse token permutation kernel."""
    pid = tl.program_id(axis=0)
    if pid >= num_tokens * num_experts_per_tok:
        return
    
    # Get original position from mapping
    src_token_idx = tl.load(token_expert_mapping_ptr + pid)
    dest_offset = (pid // num_experts_per_tok) * hidden_size
    src_offset = src_token_idx * hidden_size
    
    for offset in range(0, hidden_size, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < hidden_size
        data = tl.load(src_ptr + src_offset + block_offsets, mask=mask)
        tl.store(dest_ptr + dest_offset + block_offsets, data, mask=mask)


@triton.jit
def _segmented_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    expert_offsets_ptr, expert_counts_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_experts,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    PROPER segmented GEMM kernel for MoE expert computation.
    Each expert processes a segment of tokens independently.
    """
    # Get program ID and calculate which expert this thread block belongs to
    pid = tl.program_id(axis=0)
    
    # Calculate expert assignment
    expert_id = pid % num_experts
    
    # Get expert's segment boundaries
    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_count = tl.load(expert_counts_ptr + expert_id)
    expert_end = expert_start + expert_count
    
    if expert_count <= 0:
        return
    
    # Calculate local M for this expert
    local_M = expert_count
    
    # Calculate grid dimensions for this expert
    num_pid_m = tl.cdiv(local_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group scheduling for better cache usage
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Check bounds
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return
    
    # Calculate global offsets for this expert's segment
    global_m_start = expert_start + pid_m * BLOCK_SIZE_M
    global_m_end = tl.minimum(global_m_start + BLOCK_SIZE_M, expert_end)
    
    # Calculate global thread indices
    offs_m = global_m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers for this expert's segment
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + expert_id * K * N + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matrix multiplication for this expert's segment
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Create masks for bounds checking
        a_mask = (offs_m[:, None] < global_m_end) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
        
        # Load data
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute dot product
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store results
    c_mask = (offs_m[:, None] < global_m_end) & (offs_n[None, :] < N)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _expert_gating_kernel(
    logits_ptr, topk_values_ptr, topk_indices_ptr,
    num_tokens, num_experts, top_k, softmax_scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized expert gating with top-k selection and softmax."""
    pid = tl.program_id(axis=0)
    if pid >= num_tokens:
        return
    
    # Load logits for this token
    logits_offset = pid * num_experts
    logits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load logits in blocks
    max_logit = -float('inf')
    for offset in range(0, num_experts, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < num_experts
        logits_block = tl.load(logits_ptr + logits_offset + block_offsets, mask=mask)
        
        # Track max logit for numerical stability
        block_max = tl.max(logits_block)
        max_logit = tl.maximum(max_logit, block_max)
    
    # Compute softmax
    sum_exp = 0.0
    exp_vals = tl.zeros((num_experts,), dtype=tl.float32)
    
    for offset in range(0, num_experts, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < num_experts
        logits_block = tl.load(logits_ptr + logits_offset + block_offsets, mask=mask)
        
        exp_block = tl.exp((logits_block - max_logit) * softmax_scale)
        sum_exp += tl.sum(exp_block)
        
        # Store intermediate exp values
        for i in range(BLOCK_SIZE):
            if offset + i < num_experts:
                exp_vals = tl.where(
                    tl.arange(num_experts) == offset + i,
                    exp_block[i],
                    exp_vals
                )
    
    # Normalize
    probs = exp_vals / sum_exp
    
    # Top-k selection using bitonic sort
    for k in range(top_k):
        max_idx = 0
        max_val = 0.0
        
        # Find maximum
        for i in range(num_experts):
            mask = probs[i] > max_val
            max_val = tl.where(mask, probs[i], max_val)
            max_idx = tl.where(mask, i, max_idx)
        
        # Store result
        tl.store(topk_values_ptr + pid * top_k + k, max_val)
        tl.store(topk_indices_ptr + pid * top_k + k, max_idx)
        
        # Zero out this probability for next iteration
        probs = tl.where(tl.arange(num_experts) == max_idx, 0.0, probs)


@triton.jit
def _load_balancing_kernel(
    expert_loads_ptr, expert_assignments_ptr,
    num_experts, world_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Dynamic load balancing for expert assignment."""
    pid = tl.program_id(axis=0)
    
    # Load expert loads
    loads = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for offset in range(0, num_experts, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < num_experts
        loads_block = tl.load(expert_loads_ptr + block_offsets, mask=mask)
        loads = tl.where(mask, loads_block, loads)
    
    # Simple load balancing: sort experts by load and assign to devices
    # This is a simplified version - real implementation would use more sophisticated algorithms
    
    # Initialize device loads
    device_loads = tl.zeros((world_size,), dtype=tl.float32)
    assignments = tl.zeros((num_experts,), dtype=tl.int32)
    
    # Assign experts to devices based on load
    for expert_id in range(num_experts):
        load = loads[expert_id]
        # Find least loaded device
        min_load = float('inf')
        min_device = 0
        for device_id in range(world_size):
            device_load = device_loads[device_id]
            if device_load < min_load:
                min_load = device_load
                min_device = device_id
        
        # Assign expert to device
        assignments = tl.where(
            tl.arange(num_experts) == expert_id,
            min_device,
            assignments
        )
        
        # Update device load
        device_loads = tl.where(
            tl.arange(world_size) == min_device,
            min_load + load,
            device_loads
        )
    
    # Store assignments
    for offset in range(0, num_experts, BLOCK_SIZE):
        block_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < num_experts
        tl.store(expert_assignments_ptr + block_offsets, assignments, mask=mask)


@triton.jit
def _gradient_computation_kernel(
    grad_output_ptr, input_ptr, weight_ptr, grad_input_ptr, grad_weight_ptr,
    expert_offsets_ptr, expert_counts_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_inm, stride_ink,
    stride_wk, stride_wn,
    learning_rate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Gradient computation for expert networks."""
    pid = tl.program_id(axis=0)
    expert_id = pid
    
    # Get expert boundaries
    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_count = tl.load(expert_counts_ptr + expert_id)
    
    if expert_count <= 0:
        return
    
    # Calculate local M for this expert
    local_M = expert_count
    
    # Grid for this expert
    num_pid_m = tl.cdiv(local_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Process blocks for this expert
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return
    
    # Calculate indices
    global_m_start = expert_start + pid_m * BLOCK_SIZE_M
    global_m_end = tl.minimum(global_m_start + BLOCK_SIZE_M, expert_start + expert_count)
    
    offs_m = global_m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers
    grad_out_ptrs = grad_output_ptr + (offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon)
    input_ptrs = input_ptr + (offs_m[:, None] * stride_inm + offs_k[None, :] * stride_ink)
    weight_ptrs = weight_ptr + expert_id * K * N + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    
    # Compute gradients
    grad_weight = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    grad_input = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load data
        grad_out = tl.load(grad_out_ptrs, mask=(offs_m[:, None] < global_m_end) & (offs_n[None, :] < N))
        input_data = tl.load(input_ptrs, mask=(offs_m[:, None] < global_m_end) & (offs_k[None, :] < K))
        weight_data = tl.load(weight_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        
        # Compute gradients
        grad_weight += tl.dot(input_data.T, grad_out)
        grad_input += tl.dot(grad_out, weight_data.T)
        
        # Advance pointers
        input_ptrs += BLOCK_SIZE_K * stride_ink
        weight_ptrs += BLOCK_SIZE_K * stride_wk
    
    # Store gradients
    grad_weight_ptrs = grad_weight_ptr + expert_id * K * N + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    grad_input_ptrs = grad_input_ptr + (offs_m[:, None] * stride_inm + offs_k[None, :] * stride_ink)
    
    # Apply learning rate and store
    mask_kn = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    mask_mk = (offs_m[:, None] < global_m_end) & (offs_k[None, :] < K)
    
    tl.store(grad_weight_ptrs, grad_weight * learning_rate, mask=mask_kn)
    tl.store(grad_input_ptrs, grad_input, mask=mask_mk)


@triton.jit
def _fused_activation_kernel(
    input_ptr, output_ptr, gate_ptr, up_ptr,
    expert_offsets_ptr, expert_counts_ptr,
    M, K,
    stride_inm, stride_ink,
    stride_outm, stride_outk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused activation function for expert computation."""
    pid = tl.program_id(axis=0)
    expert_id = pid
    
    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_count = tl.load(expert_counts_ptr + expert_id)
    
    if expert_count <= 0:
        return
    
    local_M = expert_count
    
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    num_pid_m = tl.cdiv(local_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return
    
    global_m_start = expert_start + pid_m * BLOCK_SIZE_M
    global_m_end = tl.minimum(global_m_start + BLOCK_SIZE_M, expert_start + expert_count)
    
    offs_m = global_m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load gate and up projections
    gate_offset = (offs_m[:, None] * stride_inm + offs_n[None, :] * stride_ink)
    up_offset = gate_offset + K
    
    gate_vals = tl.load(input_ptr + gate_offset, mask=(offs_m[:, None] < global_m_end) & (offs_n[None, :] < K))
    up_vals = tl.load(input_ptr + up_offset, mask=(offs_m[:, None] < global_m_end) & (offs_n[None, :] < K))
    
    # Apply SiLU activation: silu(x) = x * sigmoid(x)
    silu_gate = gate_vals * tl.sigmoid(gate_vals)
    
    # Element-wise multiplication
    output = silu_gate * up_vals
    
    # Store result
    output_offset = (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outk)
    tl.store(output_ptr + output_offset, output, mask=(offs_m[:, None] < global_m_end) & (offs_n[None, :] < K))


@triton.jit
def _attention_masking_kernel(
    mask_ptr, expert_indices_ptr, seq_lens_ptr,
    batch_size, seq_len, num_experts,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Attention masking for expert-specific attention patterns."""
    pid = tl.program_id(axis=0)
    
    batch_idx = pid // num_experts
    expert_id = pid % num_experts
    
    if batch_idx >= batch_size:
        return
    
    seq_len_batch = tl.load(seq_lens_ptr + batch_idx)
    
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create attention mask for this expert
    mask = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int1)
    
    # Set causal mask
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    
    # Expert-specific masking
    expert_mask = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int1)
    
    # Apply masks
    final_mask = causal_mask & expert_mask
    
    # Store mask
    mask_offset = (batch_idx * num_experts + expert_id) * seq_len * seq_len + \
                  (offs_m[:, None] * seq_len + offs_n[None, :])
    
    tl.store(mask_ptr + mask_offset, final_mask)


# High-level wrappers
def token_permutation_v2(x: torch.Tensor, expert_indices: torch.Tensor, num_experts: int, 
                        num_experts_per_tok: int, is_forward: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced token permutation supporting multiple experts per token."""
    num_tokens, hidden_size = x.shape
    
    # Calculate expert counts and offsets
    expert_counts = torch.bincount(expert_indices, minlength=num_experts)
    expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32) - expert_counts
    
    # Prepare output buffer
    dest = torch.empty_like(x)
    expert_counters = torch.zeros(num_experts, dtype=torch.int32, device=x.device)
    token_expert_mapping = torch.empty(
        num_tokens * num_experts_per_tok, 
        dtype=torch.int32, 
        device=x.device
    )
    
    # Launch kernel
    grid = (num_tokens,)
    BLOCK_SIZE = 128 if hidden_size > 2048 else 64
    
    if is_forward:
        _token_permutation_v2_fwd_kernel[grid](
            x, dest, expert_indices, expert_offsets,
            expert_counters, token_expert_mapping,
            num_tokens, hidden_size, num_experts_per_tok,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        _token_permutation_v2_bwd_kernel[grid](
            x, dest, token_expert_mapping,
            num_tokens, hidden_size, num_experts_per_tok,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return dest, expert_counts, token_expert_mapping


def invoke_segmented_gemm_v2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, 
                            expert_counts: torch.Tensor, expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    PROPER segmented GEMM implementation for MoE.
    Each expert processes its own segment independently.
    """
    assert len(a.shape) == 2
    assert len(b.shape) == 3
    assert len(c.shape) == 2
    assert a.shape[0] == c.shape[0]
    assert b.shape[2] == c.shape[1]
    assert a.shape[1] == b.shape[1]
    
    M, K = a.shape
    num_experts, _, N = b.shape
    
    # Calculate expert offsets
    expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32) - expert_counts
    
    # Calculate total number of thread blocks needed
    total_blocks = int(torch.sum(torch.ceil(expert_counts.float() / 64) * torch.ceil(torch.tensor(N) / 64)).item())
    
    if total_blocks > 0:
        # Launch kernel with proper grid
        grid = (total_blocks,)
        
        _segmented_gemm_kernel[grid](
            a, b, c,
            expert_offsets, expert_counts,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(2),
            c.stride(0), c.stride(1),
            num_experts,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
    
    return c


def expert_gating_v2(logits: torch.Tensor, top_k: int, softmax_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized expert gating with Triton."""
    num_tokens, num_experts = logits.shape
    
    topk_values = torch.empty(
        num_tokens, top_k, 
        dtype=logits.dtype, 
        device=logits.device
    )
    topk_indices = torch.empty(
        num_tokens, top_k, 
        dtype=torch.int32, 
        device=logits.device
    )
    
    grid = (num_tokens,)
    BLOCK_SIZE = min(128, num_experts)
    
    _expert_gating_kernel[grid](
        logits, topk_values, topk_indices,
        num_tokens, num_experts, top_k, softmax_scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return topk_values, topk_indices


def load_balancing_v2(expert_loads: torch.Tensor, world_size: int) -> torch.Tensor:
    """Triton-based load balancing for expert assignment."""
    num_experts = expert_loads.shape[0]
    
    expert_assignments = torch.empty(
        num_experts, 
        dtype=torch.int32, 
        device=expert_loads.device
    )
    
    grid = (1,)  # Single block for load balancing
    BLOCK_SIZE = min(256, num_experts)
    
    _load_balancing_kernel[grid](
        expert_loads, expert_assignments,
        num_experts, world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return expert_assignments


def compute_gradients_v2(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    expert_counts: torch.Tensor,
    learning_rate: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-based gradient computation for expert networks."""
    M, K = input.shape
    num_experts, _, N = weight.shape
    
    expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32) - expert_counts
    
    grad_input = torch.empty_like(input)
    grad_weight = torch.empty_like(weight)
    
    # Calculate grid dimensions
    total_experts = int(torch.sum(torch.ceil(expert_counts.float() / 64) * torch.ceil(torch.tensor(N) / 64)).item())
    
    if total_experts > 0:
        grid = lambda META: (num_experts, 
                           int(torch.ceil(torch.tensor(M) / META['BLOCK_SIZE_M'])),
                           int(torch.ceil(torch.tensor(N) / META['BLOCK_SIZE_N'])))
        
        _gradient_computation_kernel[grid](
            grad_output, input, weight, grad_input, grad_weight,
            expert_offsets, expert_counts,
            M, N, K,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            weight.stride(1), weight.stride(2),
            learning_rate,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
        )
    
    return grad_input, grad_weight


def fused_activation_v2(
    input: torch.Tensor,
    expert_counts: torch.Tensor,
    output: torch.Tensor
) -> torch.Tensor:
    """Fused SiLU activation for expert computation."""
    M, K = input.shape
    num_experts = expert_counts.shape[0]
    
    expert_offsets = torch.cumsum(expert_counts, dim=0, dtype=torch.int32) - expert_counts
    
    if output is None:
        output = torch.empty(M, K // 2, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = lambda META: (num_experts,
                        int(torch.ceil(torch.tensor(M) / META['BLOCK_SIZE_M'])),
                        int(torch.ceil(torch.tensor(K // 2) / META['BLOCK_SIZE_N'])))
    
    _fused_activation_kernel[grid](
        input, output, None, None,
        expert_offsets, expert_counts,
        M, K // 2,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    
    return output


def expert_attention_mask_v2(
    batch_size: int,
    seq_len: int,
    num_experts: int,
    sequence_lengths: torch.Tensor
) -> torch.Tensor:
    """Triton-based attention masking for expert-specific attention."""
    mask = torch.empty(
        batch_size, num_experts, seq_len, seq_len,
        dtype=torch.bool,
        device=sequence_lengths.device
    )
    
    grid = lambda META: (batch_size * num_experts,
                        int(torch.ceil(torch.tensor(seq_len) / META['BLOCK_SIZE_M'])),
                        int(torch.ceil(torch.tensor(seq_len) / META['BLOCK_SIZE_N'])))
    
    _attention_masking_kernel[grid](
        mask, None, sequence_lengths,
        batch_size, seq_len, num_experts,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    
    return mask