
import pytest
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

def test_sequence_creation():
    token_ids = [1, 2, 3]
    sampling_params = SamplingParams(max_tokens=10)
    seq = Sequence(token_ids, sampling_params)
    assert seq.seq_id is not None
    assert seq.status == "WAITING"
    assert seq.token_ids == token_ids
    assert seq.num_prompt_tokens == len(token_ids)
    assert seq.num_cached_tokens == 0
    assert seq.block_table == []
    assert seq.temperature == sampling_params.temperature
    assert seq.max_tokens == sampling_params.max_tokens
    assert seq.ignore_eos == sampling_params.ignore_eos
    assert seq.expert_id == -1

def test_sequence_len():
    token_ids = [1, 2, 3]
    sampling_params = SamplingParams(max_tokens=10)
    seq = Sequence(token_ids, sampling_params)
    assert len(seq) == len(token_ids)

def test_sequence_append_token():
    token_ids = [1, 2, 3]
    sampling_params = SamplingParams(max_tokens=10)
    seq = Sequence(token_ids, sampling_params)
    seq.append_token(4)
    assert seq.token_ids == [1, 2, 3, 4]
