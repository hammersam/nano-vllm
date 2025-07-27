from copy import copy
from typing import Optional, List, Dict, Any
import torch
import numpy as np

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class MultiModalSequence(Sequence):
    """Extended Sequence class for handling multi-modal inputs (text + images)."""
    
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        images: Optional[List[np.ndarray]] = None,
        image_token_positions: Optional[List[int]] = None,
        image_token_ids: Optional[List[int]] = None
    ):
        super().__init__(token_ids, sampling_params)
        self.images = images or []
        self.image_token_positions = image_token_positions or []
        self.image_token_ids = image_token_ids or []
        self.vision_features: Optional[torch.Tensor] = None
        self.image_masks: Optional[torch.Tensor] = None
        
        # Track image-related metadata
        self.num_vision_tokens = 0
        self.vision_block_table = []
        
    @property
    def has_images(self) -> bool:
        """Check if this sequence contains images."""
        return len(self.images) > 0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens including both text and vision tokens."""
        return len(self.token_ids) + self.num_vision_tokens
    
    def get_image_indices(self) -> List[int]:
        """Get indices where image tokens should be inserted."""
        return self.image_token_positions
    
    def get_vision_features(self) -> Optional[torch.Tensor]:
        """Get pre-computed vision features."""
        return self.vision_features
    
    def set_vision_features(self, features: torch.Tensor):
        """Set pre-computed vision features from vision encoder."""
        self.vision_features = features
        self.num_vision_tokens = features.shape[0] if features is not None else 0
    
    def create_image_masks(self, seq_len: int) -> torch.Tensor:
        """Create attention masks for image tokens."""
        if not self.has_images:
            return torch.ones(seq_len, seq_len, dtype=torch.bool)
        
        # Create mask for vision tokens
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        
        # Mask out padding for vision tokens
        vision_start = len(self.token_ids)
        vision_end = vision_start + self.num_vision_tokens
        
        # Vision tokens can attend to all text tokens and themselves
        mask[vision_start:vision_end, :vision_start] = True
        mask[vision_start:vision_end, vision_start:vision_end] = True
        
        return mask
    
    def get_cross_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Get cross-attention mask between text and vision tokens."""
        if not self.has_images:
            return torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Text tokens can attend to vision tokens
        text_end = len(self.token_ids)
        vision_start = text_end
        vision_end = vision_start + self.num_vision_tokens
        
        # Text tokens can attend to vision tokens
        mask[:text_end, vision_start:vision_end] = True
        
        # Vision tokens can attend to text tokens
        mask[vision_start:vision_end, :text_end] = True
        
        return mask
    
    def append_vision_tokens(self, vision_token_ids: List[int]):
        """Append vision token IDs to the sequence."""
        self.image_token_ids.extend(vision_token_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sequence to dictionary for serialization."""
        base_dict = {
            'seq_id': self.seq_id,
            'status': self.status.value,
            'token_ids': self.token_ids,
            'num_prompt_tokens': self.num_prompt_tokens,
            'num_cached_tokens': self.num_cached_tokens,
            'block_table': self.block_table,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'ignore_eos': self.ignore_eos,
            'expert_id': self.expert_id,
            'images': [img.tolist() if isinstance(img, np.ndarray) else img for img in self.images],
            'image_token_positions': self.image_token_positions,
            'image_token_ids': self.image_token_ids,
            'num_vision_tokens': self.num_vision_tokens,
            'vision_block_table': self.vision_block_table,
        }
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], sampling_params: SamplingParams) -> 'MultiModalSequence':
        """Create MultiModalSequence from dictionary."""
        seq = cls(
            token_ids=data['token_ids'],
            sampling_params=sampling_params,
            images=[np.array(img) for img in data.get('images', [])],
            image_token_positions=data.get('image_token_positions', []),
            image_token_ids=data.get('image_token_ids', [])
        )
        seq.seq_id = data['seq_id']
        seq.status = SequenceStatus(data['status'])
        seq.num_prompt_tokens = data['num_prompt_tokens']
        seq.num_cached_tokens = data['num_cached_tokens']
        seq.block_table = data['block_table']
        seq.temperature = data['temperature']
        seq.max_tokens = data['max_tokens']
        seq.ignore_eos = data['ignore_eos']
        seq.expert_id = data['expert_id']
        seq.num_vision_tokens = data['num_vision_tokens']
        seq.vision_block_table = data['vision_block_table']
        return seq