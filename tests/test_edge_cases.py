import pytest
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_empty_prompt(self):
        """Test handling empty prompt"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Empty string prompt
            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("req1", "", sampling_params)
            
            # Should handle gracefully
            results = engine.step()
            # Results might be empty or contain EOS token
            assert isinstance(results, list)
    
    def test_zero_max_tokens(self):
        """Test handling zero max tokens"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Zero max tokens should be rejected
            with pytest.raises(AssertionError):
                SamplingParams(max_tokens=0)
    
    def test_very_long_prompt(self):
        """Test handling very long prompts"""
        config = Config("test_model", max_model_len=1000, num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Very long prompt (but within limits)
            long_prompt = "word " * 200  # 1000 characters
            sampling_params = SamplingParams(max_tokens=1)
            
            # Should handle within model limits
            engine.add_request("req1", long_prompt, sampling_params)
            
            # Mock tokenizer to simulate token limit
            with patch('nanovllm.engine.model_runner.get_tokenizer') as mock_get_tokenizer:
                mock_tokenizer = Mock()
                mock_tokenizer.encode.return_value = [1] * 500  # 500 tokens
                mock_get_tokenizer.return_value = mock_tokenizer
                
                results = engine.step()
                assert isinstance(results, list)
    
    def test_prompt_exceeds_model_limit(self):
        """Test prompt that exceeds model length limit"""
        config = Config("test_model", max_model_len=100, num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Mock tokenizer to return tokens exceeding limit
            with patch('nanovllm.engine.model_runner.get_tokenizer') as mock_get_tokenizer:
                mock_tokenizer = Mock()
                mock_tokenizer.encode.return_value = [1] * 200  # Exceeds 100 token limit
                mock_get_tokenizer.return_value = mock_tokenizer
                
                sampling_params = SamplingParams(max_tokens=1)
                
                # Should handle gracefully, possibly truncate or reject
                try:
                    engine.add_request("req1", "very long prompt...", sampling_params)
                    # If no exception, verify processing continues
                    results = engine.step()
                    assert isinstance(results, list)
                except Exception:
                    # Expected behavior for exceeding limits
                    pass
    
    def test_negative_temperature(self):
        """Test negative temperature values"""
        with pytest.raises(AssertionError):
            SamplingParams(temperature=-0.1)
    
    def test_temperature_zero(self):
        """Test zero temperature (deterministic)"""
        params = SamplingParams(temperature=0.0)
        assert params.temperature == 0.0
    
    def test_top_p_out_of_range(self):
        """Test top_p values outside 0-1 range"""
        with pytest.raises(AssertionError):
            SamplingParams(top_p=-0.1)
        
        with pytest.raises(AssertionError):
            SamplingParams(top_p=1.5)
    
    def test_zero_blocks_available(self):
        """Test system with zero available blocks"""
        config = Config("test_model", num_gpu_blocks=0, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager') as mock_block_manager, \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Mock block manager to have zero blocks
            mock_block_manager_instance = Mock()
            mock_block_manager_instance.get_num_free_blocks.return_value = 0
            mock_block_manager.return_value = mock_block_manager_instance
            
            engine = LLMEngine(config)
            
            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("req1", "test", sampling_params)
            
            # Should handle gracefully, possibly queue or reject
            results = engine.step()
            assert isinstance(results, list)
    
    def test_duplicate_request_ids(self):
        """Test handling duplicate request IDs"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            sampling_params = SamplingParams(max_tokens=1)
            
            # Add first request
            engine.add_request("duplicate_id", "test1", sampling_params)
            
            # Try to add second request with same ID
            # Should handle gracefully (replace or reject)
            try:
                engine.add_request("duplicate_id", "test2", sampling_params)
                # If no exception, verify behavior
                results = engine.step()
                assert isinstance(results, list)
            except Exception:
                # Expected if duplicate IDs are not allowed
                pass
    
    def test_special_characters_in_prompt(self):
        """Test handling special characters in prompts"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            special_prompts = [
                "",
                "\n\t\r",
                "unicode: 你好世界",
                "emoji: 😀🎉",
                "html: <div>test</div>",
                "code: print('hello')",
                "math: 1+1=2",
                "mixed: Hello\nWorld\t!"
            ]
            
            sampling_params = SamplingParams(max_tokens=1)
            
            for i, prompt in enumerate(special_prompts):
                try:
                    engine.add_request(f"req{i}", prompt, sampling_params)
                except Exception as e:
                    # Should handle gracefully
                    assert isinstance(e, (UnicodeError, ValueError, TypeError))
    
    def test_max_tokens_boundary(self):
        """Test max tokens at boundary values"""
        config = Config("test_model", max_model_len=1000, num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Test max tokens equal to model limit
            max_possible = 1000 - 1  # Leave room for prompt
            sampling_params = SamplingParams(max_tokens=max_possible)
            
            engine.add_request("req1", "short", sampling_params)
            
            # Should handle boundary value
            results = engine.step()
            assert isinstance(results, list)
    
    def test_sequence_with_zero_tokens(self):
        """Test sequence with zero tokens"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(max_tokens=1)
        seq = Sequence([], sampling_params)
        
        assert seq.num_prompt_tokens == 0
        assert len(seq) == 0
        assert seq.num_cached_tokens == 0
    
    def test_block_table_edge_cases(self):
        """Test block table edge cases"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(max_tokens=1)
        seq = Sequence([1, 2, 3], sampling_params)
        
        # Empty block table
        assert seq.block_table == []
        
        # Adding blocks
        seq.block_table.extend([0, 1, 2])
        assert seq.block_table == [0, 1, 2]
    
    def test_memory_exhaustion_recovery(self):
        """Test system recovery from memory exhaustion"""
        config = Config("test_model", num_gpu_blocks=5, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager') as mock_block_manager, \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Simulate memory exhaustion
            mock_block_manager_instance = Mock()
            mock_block_manager_instance.get_num_free_blocks.return_value = 0
            mock_block_manager_instance.allocate.side_effect = [
                None,  # First allocation succeeds
                RuntimeError("Out of memory"),  # Second fails
            ]
            mock_block_manager.return_value = mock_block_manager_instance
            
            engine = LLMEngine(config)
            
            sampling_params = SamplingParams(max_tokens=1)
            
            # Add requests that might exhaust memory
            for i in range(3):
                try:
                    engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
                except Exception:
                    # Should handle gracefully
                    pass
            
            # System should remain stable
            results = engine.step()
            assert isinstance(results, list)
    
    def test_concurrent_abort_and_add(self):
        """Test handling concurrent abort and add operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            sampling_params = SamplingParams(max_tokens=1)
            
            # Add initial requests
            for i in range(5):
                engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
            
            # Concurrent operations
            engine.abort_request("req2")
            engine.add_request("req6", "new prompt", sampling_params)
            engine.abort_request("req0")
            
            # System should handle gracefully
            results = engine.step()
            assert isinstance(results, list)
    
    def test_invalid_unicode_handling(self):
        """Test handling invalid Unicode sequences"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Test various invalid Unicode scenarios
            invalid_cases = [
                "\x80\x81",  # Invalid UTF-8
                "\ud800",    # Unpaired surrogate
                "\udc00",    # Unpaired surrogate
                "\uffff",    # Non-character
            ]
            
            sampling_params = SamplingParams(max_tokens=1)
            
            for i, prompt in enumerate(invalid_cases):
                try:
                    engine.add_request(f"req{i}", prompt, sampling_params)
                    results = engine.step()
                    assert isinstance(results, list)
                except (UnicodeError, ValueError):
                    # Expected for invalid Unicode
                    pass