from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self._num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.expert_id = -1  # Track associated expert for MoE optimization

    def __len__(self):
        return len(self.token_ids)

    def __lt__(self, other):
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, num_cached_tokens):
        assert num_cached_tokens % self.block_size == 0
        self._num_cached_tokens = num_cached_tokens

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_token(self):
        return self.token_ids[-1]

    def block(self, i):
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def last_block(self):
        n = self.num_blocks
        return self.token_ids[(n-1)*self.block_size:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
