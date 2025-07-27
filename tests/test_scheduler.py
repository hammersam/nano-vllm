
import pytest
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.config import Config


def test_scheduler_add():
    config = Config("test_model")
    scheduler = Scheduler(config)
    assert scheduler.is_finished()

    sampling_params = SamplingParams(max_tokens=10)
    seq = Sequence([1, 2, 3], sampling_params)
    scheduler.add(seq)

    assert not scheduler.is_finished()
    assert len(scheduler.waiting) == 1
