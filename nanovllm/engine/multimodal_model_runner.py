import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np

from nanovllm.config import Config
from nanovllm.engine.multimodal_sequence import MultiModalSequence
from nanovllm.layers.vision import VisionEncoder, VisionProcessor
from nanovllm.layers.cross_attention import CrossModalAttention
from nanovllm.engine.model_runner import ModelRunner


class MultiModalModelRunner(ModelRunner):
    """Extended ModelRunner with multi-modal inference capabilities."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Initialize vision components if multi-modal is enabled
        if config.enable_multimodal:
            self.vision_encoder = VisionEncoder(
                model_name=config.vision_model,
                hidden_size=config.hf_config.hidden_size,
                image_size=config.max_image_size,
                num_patches=config.num_vision_tokens
            )
            self.vision_processor = VisionProcessor(self.vision_encoder)
            self.cross_attention = CrossModalAttention(
                hidden_size=config.hf_config.hidden_size,
                num_attention_heads=config.hf_config.num_attention_heads
            )
            
            # Move vision encoder to GPU
            self.vision_encoder.to_device("cuda")
            
            # Cache for vision features
            self.vision_cache = {}
        else:
            self.vision_encoder = None
            self.vision_processor = None
            self.cross_attention = None
            self.vision_cache = None
    
    def process_vision_inputs(
        self,
        seqs: List[MultiModalSequence]
    ) -> List[Tuple[torch.Tensor, int]]:
        """Process vision inputs for multi-modal sequences."""
        if not self.vision_processor:
            return [(None, 0) for _ in seqs]
        
        results = []
        for seq in seqs:
            if seq.has_images:
                # Process images
                vision_features, num_vision_tokens = self.vision_processor.process_images(
                    seq.images
                )
                
                # Cache vision features
                if vision_features is not None:
                    vision_features = vision_features.to("cuda")
                    seq.set_vision_features(vision_features.flatten(0, 1))
                
                results.append((vision_features, num_vision_tokens))
            else:
                results.append((None, 0))
        
        return results
    
    def prepare_multimodal_inputs(
        self,
        seqs: List[MultiModalSequence],
        is_prefill: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare inputs for multi-modal inference including vision features.
        
        Args:
            seqs: List of multi-modal sequences
            is_prefill: Whether this is prefill phase
            
        Returns:
            Tuple of (input_ids, positions, vision_features, attention_masks)
        """
        # Process vision inputs
        vision_results = self.process_vision_inputs(seqs)
        
        # Prepare basic inputs using parent class
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        
        # Prepare vision features and masks
        vision_features_list = []
        attention_masks_list = []
        
        for seq, (vision_feat, num_vision_tokens) in zip(seqs, vision_results):
            if vision_feat is not None:
                vision_features_list.append(seq.get_vision_features())
                
                # Create attention mask
                total_len = len(seq) + num_vision_tokens
                attention_mask = seq.create_image_masks(total_len)
                attention_masks_list.append(attention_mask)
            else:
                vision_features_list.append(None)
                attention_masks_list.append(None)
        
        # Stack vision features if any
        if any(vf is not None for vf in vision_features_list):
            max_vision_len = max(
                vf.shape[0] if vf is not None else 0 
                for vf in vision_features_list
            )
            
            # Pad vision features
            padded_vision_features = []
            for vf in vision_features_list:
                if vf is not None:
                    padding_len = max_vision_len - vf.shape[0]
                    if padding_len > 0:
                        padding = torch.zeros(
                            padding_len, vf.shape[1], 
                            device=vf.device, dtype=vf.dtype
                        )
                        vf = torch.cat([vf, padding], dim=0)
                    padded_vision_features.append(vf)
                else:
                    padded_vision_features.append(
                        torch.zeros(max_vision_len, self.config.hf_config.hidden_size, device="cuda")
                    )
            
            vision_features = torch.stack(padded_vision_features)
        else:
            vision_features = None
        
        return input_ids, positions, vision_features, attention_masks_list
    
    def forward_multimodal(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        vision_features: Optional[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass with multi-modal processing."""
        if vision_features is None or self.cross_attention is None:
            # Standard forward pass without vision
            return self.model(input_ids, positions)
        
        # Get text embeddings
        text_hidden_states = self.model.embed_tokens(input_ids)
        
        # Apply cross-attention between text and vision
        batch_size = text_hidden_states.shape[0]
        updated_hidden_states = []
        
        for i in range(batch_size):
            text_feat = text_hidden_states[i:i+1]  # [1, seq_len, hidden_size]
            vision_feat = vision_features[i:i+1]   # [1, vision_len, hidden_size]
            
            # Apply cross-attention
            updated_feat = self.cross_attention(
                text_feat,
                vision_feat,
                attention_mask=attention_masks[i] if attention_masks else None
            )
            
            updated_hidden_states.append(updated_feat.squeeze(0))
        
        # Stack updated features
        updated_hidden_states = torch.stack(updated_hidden_states)
        
        # Continue with model forward pass
        return self.model.layers_forward(updated_hidden_states, positions)
    
    def run(self, seqs: List[MultiModalSequence], is_prefill: bool) -> List[int]:
        """Override run method to handle multi-modal sequences."""
        if not self.config.enable_multimodal:
            # Use parent implementation for text-only
            return super().run(seqs, is_prefill)
        
        # Prepare multi-modal inputs
        input_ids, positions, vision_features, attention_masks = self.prepare_multimodal_inputs(
            seqs, is_prefill
        )
        
        # Prepare sampling parameters
        temperatures = self.prepare_sample(seqs)
        
        # Forward pass with multi-modal processing
        with torch.no_grad():
            if vision_features is not None and self.cross_attention is not None:
                logits = self.forward_multimodal(
                    input_ids, positions, vision_features, attention_masks
                )
            else:
                logits = self.run_model(input_ids, positions, is_prefill)
        
        # Sample tokens
        token_ids = self.sampler(logits, temperatures).tolist()
        
        return token_ids
    
    def clear_vision_cache(self):
        """Clear the vision feature cache."""
        if self.vision_cache is not None:
            self.vision_cache.clear()
    
    def get_vision_stats(self) -> dict:
        """Get statistics about vision processing."""
        if not self.config.enable_multimodal:
            return {}
        
        return {
            "vision_cache_size": len(self.vision_cache) if self.vision_cache else 0,
            "vision_encoder_device": str(self.vision_encoder.device) if self.vision_encoder else "N/A",
            "cross_attention_enabled": self.cross_attention is not None
        }