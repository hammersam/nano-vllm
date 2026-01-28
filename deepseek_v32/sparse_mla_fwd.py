# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


# sparse_mla_fwd为kernel准备好各种参数
@tilelang.jit(
    # main最后两个参数作为返回值
    out_idx=[-2, -1],
    pass_configs={
        # 禁止自动使用tma和warp specialization
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (topk %
            block_I == 0), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    # `kv_group`表示kv head的数量，一个kv head要对应`head_kv`个query head
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    # dsa中使用的是mqa(i.e., kv_group = 1)，但是如果kv_group != 1，
    # 则对应gqa场景，不同group的queries可以有不同的topk keys
    # indices是与kv head绑定的，而不是与query head绑定，只需为每个
    # kv group准备一份即可，同一组内的所有query heads共享一套indices
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert (
            kv_group == 1
        ), "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        # 如果一个kv head要对应过多的query head(>64)，或者说一个thread block
        # 要处理过多的query head，硬件资源压力会过大，因此划分成多组/多个block进行计算
        # 每组中每个kv head最多对应64个kv head，一共有`REPLICATE_H`组
        REPLICATE_H = head_kv // 64
    else:
        # 如果一个kv head对应query head的数量小于等于64，直接使用一个thread block
        # 进行处理即可，无需进行划分
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    # 这是实际的tilelang kernel
    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            cache_len: T.int32, # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
            Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
                # 原本一个thread block需要处理的query head被划分成`REPLICATE_H`个
                # block进行处理
                seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
                ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            # sum of exps for whole topk or current BI
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i + cache_len

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # ceil(topk / BI) = NI
            for i_i in T.Pipelined(NI, num_stages=num_stages):

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                              d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i,
                                                  D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    # softmax(((q @ k) - m) * sm_scale)
                    # [exp((s0 - m_pre) * sm_scale), s1, s2, s3] * exp((m_pre - m_i) * sm_scale) ->
                    # [exp((s0 - m_i) * sm_scale), ...]
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                # 每个warp负责完整的一行，不用进行跨warp的同步和通信
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            # 先拷贝到shared mem中重排数据布局(从tensor core的swizzled转为线性)以及
            # 数据类型的转换(例如acc_o从float转为bf16)，合并对global mem的内存访问
            # (memory coalescing)，提高显存带宽利用率
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(Lse_shared, Lse[b_i, s_i, H0:H1])

    return main


# python wrapper of sparse mla tilelang kernel
def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256
):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    assert seq_len_kv >= seq_len

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads
    )
    
    out, lse = kernel(q, kv, indices, seq_len_kv - seq_len)
    
    return out, lse


# 用pytorch实现的sparse mla参考版本
def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, cache_len=None):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    # dim_q也叫做dim_plus_tail_dim
    # q与kv维度相同
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    assert sk >= sq

    if cache_len is None:
        cache_len = sk - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    # 保证q,k之间的causal关系
    causal_mask = (torch.arange(
        0, sq, dtype=torch.int32, device="cuda") + cache_len).view(-1, 1) >= torch.arange(
            0, sk, dtype=torch.int32, device="cuda").view(1, -1)

    # 将被选中的key标记为True，添加一个"垃圾箱"维度防止索引越界
    mask = q.new_zeros(b, g, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    # 去除掉"垃圾箱"维度
    mask = mask[..., :-1]
    # topk && causal masking
    mask = mask & causal_mask.view(1, 1, sq, sk)
    # 为了方便之后和scores的运算
    mask = mask.view(b, g, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    # (b, g, h_index, sq, sk)
    p = score.softmax(dim=-1)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=64,
    num_stages=2,
    threads=256
):
    torch.random.manual_seed(0)

    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")

    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                # 生成0到max(1,t)的随机排列
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    tl_out, tl_lse = sparse_mla_fwd_interface(
        q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(
            q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd kvcache io bandwidth(tb/s) = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    # Q @ K.T and Score @ V
    print("fwd tflop/s = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        num_stages=2,
        threads=256
    )
