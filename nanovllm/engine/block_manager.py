from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


def compute_hash(token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        assert hash != -1
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return f"{(self.block_id, self.ref_count, self.hash)}"


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int, expert_id: int = -1):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        block.expert_id = expert_id  # Track associated expert for MoE
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # A map from a block's content hash to the physical block ID that stores the
            # corresponding KV cache. This is the core of our prefix caching mechanism.
            block_id = self.hash_to_block_id.get(h, -1)
            
            # MoE optimization: Prefer blocks from the same expert
            preferred_block_id = -1
            if hasattr(seq, 'expert_id') and seq.expert_id != -1:
                for bid in self.used_block_ids:
                    if self.blocks[bid].expert_id == seq.expert_id and self.blocks[bid].hash == h:
                        preferred_block_id = bid
                        break
            
            block_id = preferred_block_id if preferred_block_id != -1 else block_id
            
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                # Allocate with expert affinity if available
                expert_id = getattr(seq, 'expert_id', -1)
                block = self._allocate_block(block_id, expert_id)
            else:
                seq.num_cached_tokens += self.block_size
                block = self.blocks[block_id]
                if block.ref_count > 0:
                    # The cached block is actively in use by other sequences.
                    assert block_id in self.used_block_ids
                    block.ref_count += 1
                else:
                    # This block was previously used but has since been freed (ref_count
                    # dropped to 0) and returned to the free pool. We now "re-allocate"
                    # it by moving it from the free pool back to the used pool and
                    # setting its reference count to 1.
                    assert block_id in self.free_block_ids
                    expert_id = getattr(seq, 'expert_id', -1)
                    block = self._allocate_block(block_id, expert_id)
            # If the hash is valid (i.e., the block is full), we "seal" the block by
            # updating its metadata and adding it to the cache map for future reuse.
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        """
        Checks if a new token can be appended to the sequence.

        This is a non-destructive check used by the scheduler. A new physical
        block is only required when the sequence length is `N * block_size + 1`,
        signifying that the previous token just started a new conceptual block.
        """
        # The boolean result of the comparison is cast to an integer (0 or 1).
        # We only need a free block if the sequence is about to allocate one.
        num_required_blocks = 1 if len(seq) % self.block_size == 1 else 0
        return len(self.free_block_ids) >= num_required_blocks

    def may_append(self, seq: Sequence):
        """
        Performs memory operations for a sequence before the next token is generated.

        This is a destructive action that either allocates a new block or
        "seals" a full block for future prefix caching. It's called after
        `can_append` has confirmed that allocation is possible.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # Case 1: The sequence has just spilled into a new conceptual block.
        # We must allocate a new physical block for the next token's KV cache.
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1, "The previous block should have been sealed."
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # Case 2: The sequence has just perfectly filled a block.
        # We "seal" this block by computing its hash, making it available for caching.
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1, "Block should not be sealed yet."
            token_ids = seq.last_block()
            # prefix caching
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # Case 3: The last block still has free slots. Do nothing.
        else:
            assert last_block.hash == -1, "Block should not be sealed if it's not full."
