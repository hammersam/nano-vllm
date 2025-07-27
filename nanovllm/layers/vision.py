import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(nn.Module):
    """Vision encoder for processing images in multi-modal inference."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        hidden_size: int = 768,
        num_patches: int = 49,
        patch_size: int = 32,
        image_size: int = 224,
        freeze_weights: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.freeze_weights = freeze_weights
        
        # Load pre-trained vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Freeze vision model weights if specified
        if self.freeze_weights:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Projection layer to match text embedding dimension
        self.vision_projection = nn.Linear(
            self.vision_model.config.hidden_size,
            hidden_size
        )
        
        # Learnable vision token type embedding
        self.vision_token_type_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_size) * 0.02
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision encoding.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            
        Returns:
            Vision features [batch_size, num_patches, hidden_size]
        """
        # Get vision features from CLIP
        vision_outputs = self.vision_model(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state
        
        # Remove CLS token and reshape
        vision_features = vision_features[:, 1:]  # Remove CLS token
        
        # Project to target dimension
        vision_features = self.vision_projection(vision_features)
        
        # Add vision token type embedding
        vision_features = vision_features + self.vision_token_type_embedding
        
        return vision_features
    
    def preprocess_images(
        self,
        images: List[np.ndarray],
        return_tensors: str = "pt"
    ) -> torch.Tensor:
        """
        Preprocess images for vision encoder.
        
        Args:
            images: List of numpy arrays representing images
            return_tensors: Format to return tensors in
            
        Returns:
            Preprocessed image tensor
        """
        # Ensure images are RGB and correct size
        processed_images = []
        for img in images:
            if len(img.shape) == 2:
                # Grayscale to RGB
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                # RGBA to RGB
                img = img[..., :3]
            processed_images.append(img)
        
        # Use CLIP processor
        inputs = self.image_processor(
            images=processed_images,
            return_tensors=return_tensors
        )
        
        return inputs['pixel_values']
    
    def get_vision_token_count(self) -> int:
        """Get the number of vision tokens per image."""
        return self.num_patches
    
    @property
    def device(self):
        """Get the device of the vision encoder."""
        return next(self.parameters()).device
    
    def to_device(self, device: torch.device):
        """Move vision encoder to specified device."""
        self.vision_model = self.vision_model.to(device)
        self.vision_projection = self.vision_projection.to(device)
        return self


class VisionProcessor:
    """Utility class for handling vision preprocessing and postprocessing."""
    
    def __init__(self, vision_encoder: VisionEncoder):
        self.vision_encoder = vision_encoder
        self.image_placeholder_token = "<|image|>"
        
    def process_images(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, int]:
        """
        Process images and return vision features along with token count.
        
        Args:
            images: List of images to process
            
        Returns:
            Tuple of (vision_features, num_vision_tokens)
        """
        if not images:
            return None, 0
            
        # Preprocess images
        image_tensor = self.vision_encoder.preprocess_images(images)
        
        # Get vision features
        with torch.no_grad():
            vision_features = self.vision_encoder(image_tensor)
        
        # Get token count
        num_vision_tokens = self.vision_encoder.get_vision_token_count() * len(images)
        
        return vision_features, num_vision_tokens
    
    def replace_image_placeholders(
        self,
        text: str,
        images: List[np.ndarray]
    ) -> Tuple[str, List[int]]:
        """
        Replace image placeholders with actual image tokens.
        
        Args:
            text: Input text with image placeholders
            images: List of images to embed
            
        Returns:
            Tuple of (modified_text, image_token_positions)
        """
        if not images:
            return text, []
        
        # Count image placeholders
        placeholder_count = text.count(self.image_placeholder_token)
        if placeholder_count != len(images):
            raise ValueError(
                f"Number of images ({len(images)}) must match placeholders ({placeholder_count})"
            )
        
        # Replace placeholders with image tokens
        modified_text = text
        image_token_positions = []
        
        for i, _ in enumerate(images):
            pos = modified_text.find(self.image_placeholder_token)
            image_token_positions.append(pos)
            modified_text = modified_text.replace(
                self.image_placeholder_token,
                f"<|vision_{i}|>",  # Special token for image
                1
            )
        
        return modified_text, image_token_positions
    
    def get_image_token_ids(self, num_images: int) -> List[int]:
        """Generate token IDs for image tokens."""
        # Use special token IDs for vision tokens
        base_vision_token_id = 100000  # Arbitrary large number for vision tokens
        token_ids = []
        
        for img_idx in range(num_images):
            start_id = base_vision_token_id + img_idx * 1000
            num_tokens = self.vision_encoder.get_vision_token_count()
            token_ids.extend(range(start_id, start_id + num_tokens))
        
        return token_ids