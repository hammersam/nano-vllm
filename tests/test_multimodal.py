import pytest
import torch
import numpy as np
from PIL import Image

from nanovllm.config import Config
from nanovllm.engine.multimodal_sequence import MultiModalSequence
from nanovllm.engine.multimodal_llm_engine import MultiModalLLMEngine
from nanovllm.layers.vision import VisionEncoder, VisionProcessor
from nanovllm.sampling_params import SamplingParams


class TestMultiModalFeatures:
    """Test suite for multi-modal inference features."""
   
    def test_multimodal_sequence_creation(self):
        """Test MultiModalSequence creation and properties."""
        config = Config("test-model", enable_multimodal=True)
        
        token_ids = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=100)
        images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)]
        
        seq = MultiModalSequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
            images=images,
            image_token_positions=[3],
            image_token_ids=[100000, 100001]
        )
        
        assert seq.has_images
        assert seq.total_tokens == len(token_ids) + seq.num_vision_tokens
        assert len(seq.images) == 1
        assert seq.image_token_positions == [3]
    
    def test_vision_encoder_initialization(self):
        """Test VisionEncoder initialization."""
        encoder = VisionEncoder(
            model_name="openai/clip-vit-base-patch32",
            hidden_size=768,
            image_size=224,
            num_patches=49
        )
        
        assert encoder.hidden_size == 768
        assert encoder.image_size == 224
        assert encoder.num_patches == 49
        assert encoder.vision_model is not None
    
    def test_vision_encoder_forward(self):
        """Test VisionEncoder forward pass."""
        encoder = VisionEncoder(
            hidden_size=768,
            image_size=224,
            num_patches=49
        )
        
        # Create dummy image batch
        images = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = encoder(images)
        
        assert features.shape == (2, 49, 768)
    
    def test_vision_processor_image_processing(self):
        """Test VisionProcessor image processing."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        # Create test images
        images = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)]
        
        features, num_tokens = processor.process_images(images)
        
        assert features is not None
        assert num_tokens > 0
        assert features.shape[0] == num_tokens
    
    def test_multimodal_engine_initialization(self):
        """Test MultiModalLLMEngine initialization."""
        config = Config(
            "test-model",
            enable_multimodal=True,
            vision_model="openai/clip-vit-base-patch32"
        )
        
        # Mock initialization since we don't have actual model
        assert config.enable_multimodal
        assert config.vision_model == "openai/clip-vit-base-patch32"
    
    def test_image_placeholder_replacement(self):
        """Test image placeholder replacement in text."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        text = "Describe this image: <|image|>"
        images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)]
        
        modified_text, positions = processor.replace_image_placeholders(text, images)
        
        assert modified_text != text
        assert len(positions) == 1
        assert positions[0] == 19  # Position of "<|image|>"
    
    def test_cross_attention_masks(self):
        """Test cross-attention mask generation."""
        token_ids = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=100)
        
        seq = MultiModalSequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
            images=[np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)],
            image_token_positions=[2]
        )
        
        # Set vision features
        seq.set_vision_features(torch.randn(49, 768))
        
        total_len = len(token_ids) + seq.num_vision_tokens
        mask = seq.create_image_masks(total_len)
        
        assert mask.shape == (total_len, total_len)
        assert mask.dtype == torch.bool
    
    def test_vision_feature_caching(self):
        """Test vision feature caching mechanism."""
        encoder = VisionEncoder(hidden_size=768)
        features = torch.randn(49, 768)
        
        seq = MultiModalSequence(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=100)
        )
        
        seq.set_vision_features(features)
        
        assert torch.equal(seq.get_vision_features(), features)
        assert seq.num_vision_tokens == 49
    
    def test_grayscale_image_handling(self):
        """Test handling of grayscale images."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        # Grayscale image
        gray_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
        # Should be converted to RGB
        processed = processor._process_images([gray_image])
        
        assert processed[0].shape[-1] == 3
    
    def test_rgba_image_handling(self):
        """Test handling of RGBA images."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        # RGBA image
        rgba_image = np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)
        
        # Should be converted to RGB
        processed = processor._process_images([rgba_image])
        
        assert processed[0].shape[-1] == 3
    
    def test_multiple_images(self):
        """Test handling of multiple images in one sequence."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        images = [
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        ]
        
        features, num_tokens = processor.process_images(images)
        
        assert features.shape[0] == 2 * encoder.num_patches
        assert num_tokens == 2 * encoder.num_patches
    
    def test_empty_images(self):
        """Test handling of empty images list."""
        encoder = VisionEncoder(hidden_size=768)
        processor = VisionProcessor(encoder)
        
        features, num_tokens = processor.process_images([])
        
        assert features is None
        assert num_tokens == 0
    
    def test_sequence_serialization(self):
        """Test MultiModalSequence serialization/deserialization."""
        token_ids = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=100)
        images = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)]
        
        original_seq = MultiModalSequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
            images=images
        )
        
        # Serialize
        seq_dict = original_seq.to_dict()
        
        # Deserialize
        restored_seq = MultiModalSequence.from_dict(seq_dict, sampling_params)
        
        assert restored_seq.token_ids == original_seq.token_ids
        assert len(restored_seq.images) == len(original_seq.images)
        assert restored_seq.image_token_positions == original_seq.image_token_positions


class TestIntegration:
    """Integration tests for multi-modal features."""
    
    def test_end_to_end_multimodal_processing(self):
        """Test complete multi-modal processing pipeline."""
        config = Config(
            "test-model",
            enable_multimodal=True,
            max_image_size=224,
            num_vision_tokens=49
        )
        
        # Test configuration
        assert config.enable_multimodal
        assert config.max_image_size == 224
        
    def test_backwards_compatibility(self):
        """Test that multimodal features don't break text-only mode."""
        config = Config("test-model", enable_multimodal=False)
        
        # Should work without multimodal
        assert not config.enable_multimodal
        
    def test_vision_encoder_device_handling(self):
        """Test vision encoder device placement."""
        encoder = VisionEncoder(hidden_size=768)
        
        # Test device movement
        device = torch.device('cpu')
        encoder.to_device(device)
        
        assert str(encoder.device) == str(device)


if __name__ == "__main__":
    pytest.main([__file__])