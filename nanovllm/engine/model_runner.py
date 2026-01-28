# model_runner.py连接了Scheduler和底层模型，核心函数run封装了single step inference所有逻辑
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


# 负责模型实际执行和gpu资源管理，主要职责是模型加载、显存分配(特别是kvcache)、
# 管理多卡通信(tensor parallelism)，并执行模型的forward pass
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化分布式环境
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载模型权重到GPU
        load_model(self.model, config.model)
        # 初始化采样器
        self.sampler = Sampler()
        self.warmup_model()
        # 计算GPU剩余显存，根据block size预先分配用于paged attn的kv cache显存池
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # 为了加速decode阶段，预先录制不同batch size下的cuda执行图，减少cpu启动kernel的开销
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # 基于SharedMemory实现多进程通信(IPC)，通过一些自定义的RPC机制(loop, read_shm, write_shm, call)，
            # 允许主进程(rank 0)向其他进程发送指令(e.g., "run", "exit")，协调多卡并行推理
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # (peak - current)表示模型推理过程中需要的临时显存空间
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个block负责多少token的kvcache
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 提前规定好多少显存可以用作kvcache，将这些划分成一个个block，
        # 根据`block_bytes`计算出可以划分成多少个block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # pre-allocate kvcache memory pool, managed by `BlockManager`
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 给每个attention layer绑定在gpu物理显存中分配好的kvcache
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # (num_seqs, max_len)
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # 将逻辑上的Sequence对象转化为模型需要的Tensor格式(input_ids, positions...)，
    # 构建PagedAttention需要的metadata(e.g., block_tables, slot_mapping)
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # 需要处理的序列token累积长度，被缓存的token不包含在内
        cu_seqlens_q = [0]
        # 需要关注的总上下文长度(历史缓存 + 新token)
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 记录一下这些要被计算的token在原先序列中的逻辑位置
            # 如果有缓存，位置从num_cached_tokens开始，否则从0开始
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                # 记录没有被缓存token在kvcache中的位置
                slot_mapping.extend(list(range(start, end)))
        # 如果存在cached token，需要block_tables才能找到分散在
        # 显存中的kv blocks
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 传递slot_mapping和block_tables
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            # 永远从最后一个位置开始
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # phy_blk_id * blk_size + offset_within_blk
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 计算发生的地方，性能优化的关键点
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # prefill阶段计算量大，直接运行模型
            # prefill或者large batch使用eager mode
            # 传递`positions`用于计算位置编码
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # decode阶段计算量小但是频次高，受限于cpu发射kernel的延迟，因此
            # 使用cuda graph(graph.replay())来消除cpu开销，提升生成速度
            bs = input_ids.size(0)
            context = get_context()
            # 从self.graph_bs中找到第一个大于或等于bs的bucket
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 只拷贝bs个样本，后面用作padding
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放录制好的cuda graph，消除cpu launch kernel的开销
            graph.replay()
            # 只获取前bs个结果
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 最核心的入口函数，是LLMEngine驱动模型运转的直接接口，串联了推理的完整生命周期
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # prefill情况下input_ids包含着当前batch中所有序列需要
        # 计算的token(去除掉已缓存部分)，decode阶段包含着每个序列
        # 最新生成的一个token，因此shape应该是(total_new_tokens/batch_size,)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # (num_tokens, vocab_size)
        logits = self.run_model(input_ids, positions, is_prefill)
        # 调用sampler，根据logits和温度参数选择下一个token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置上下文
        reset_context()
        return token_ids

    # 为了避免在模型decode阶段cpu launch kernel的时间比gpu实际计算
    # 的时间还要长，使用CUDA Graph将一连串的kernel launch录制下来，
    # 打包成一个graph，之后推理时cpu只需要发送一条指令，gpu就能够自动
    # 按顺序执行graph内部多个kernel，可以减少cpu与gpu交互次数，提升推理速度
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义一组"档位"(buckets)，因为没办法为每一个可能的batch size(e.g., 1 to 512)
        # 都单独录制一个graph，太耗显存了
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # CUDA Graph需要一部份显存来存放中间激活值，反向遍历graph_bs，
        # 复用录制最大batch size时分配的内存池
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup: 为了让pytorch完成所有懒加载初始化，确保显存分配器准备好，
            # 录制过程不会出现非预期的内存分配
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                # capture: 使用graph对象和内存池graph_pool进入录制上下文，
                # with block中所有gpu操作(kernel launch)都不会被立即执行，
                # 而是会被记录到graph中
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    
            # 第一次录制(i.e., largest batch size)，保存内存池供后续较小的batch size复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            # 同步并重置上下文，准备下一次循环
            torch.cuda.synchronize()
            reset_context()

        # graph所使用的一些固定tensor，之后decode阶段需要把新的数据拷贝进去
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
