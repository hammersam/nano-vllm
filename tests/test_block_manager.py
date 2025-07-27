
import pytest
from nanovllm.engine.block_manager import BlockManager
from nanovllm.config import Config


def test_block_manager_allocation():
    config = Config("test_model", kvcache_block_size=16, num_gpu_blocks=100)
    block_manager = BlockManager(config)

    assert block_manager.get_num_free_blocks() == 100

    # Allocate some blocks
    block_manager.allocate(5)
    assert block_manager.get_num_free_blocks() == 95

    # Free the blocks
    block_manager.free(5)
    assert block_manager.get_num_free_blocks() == 100
