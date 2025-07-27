import numpy as np
from typing import List, Optional, Union, Dict, Any
from PIL import Image

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.multimodal_sequence import MultiModalSequence
from nanovllm.engine.multimodal_model_runner import MultiModalModelRunner
from nanovllm.engine.llm_engine import LLMEngine


class MultiModalLLMEngine(LLMEngine):
    """Extended LLMEngine with multi-modal inference capabilities."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        
        # Replace standard model runner with multi-modal version
        if self.config.enable_multimodal:
            self.model_runner = MultiModalModelRunner(self.config)
    
    def add_request(
        self,
        prompt: str | List[int],
        sampling_params: SamplingParams,
        images: Optional[List[Union[str, np.ndarray, Image.Image]]] = None
    ):
        """
        Add a request with optional images for multi-modal inference.
        
        Args:
            prompt: Text prompt or token IDs
            sampling_params: Sampling parameters
            images: Optional list of images (file paths, numpy arrays, or PIL Images)
        """
        if isinstance(prompt, str):
            # Handle text prompts with potential image placeholders
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        
        # Process images if provided
        processed_images = None
        image_token_positions = None
        image_token_ids = None
        
        if images and self.config.enable_multimodal:
            processed_images = self._process_images(images)
            
            # Find image token positions in prompt
            if isinstance(prompt, str):
                image_token_positions = self._find_image_tokens(prompt)
                image_token_ids = self._generate_image_token_ids(len(processed_images))
        
        # Create multi-modal sequence
        seq = MultiModalSequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
            images=processed_images,
            image_token_positions=image_token_positions,
            image_token_ids=image_token_ids
        )
        
        self.scheduler.add(seq)
    
    def _process_images(
        self,
        images: List[Union[str, np.ndarray, Image.Image]]
    ) -> List[np.ndarray]:
        """Process and normalize images for multi-modal inference."""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                # Load from file path
                img_array = np.array(Image.open(img))
            elif isinstance(img, Image.Image):
                # Convert PIL Image to numpy
                img_array = np.array(img)
            elif isinstance(img, np.ndarray):
                # Use as-is
                img_array = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Ensure correct shape and format
            if len(img_array.shape) == 2:
                # Grayscale to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                # RGBA to RGB
                img_array = img_array[..., :3]
            
            # Resize if needed
            img_array = self._resize_image(img_array)
            processed_images.append(img_array)
        
        return processed_images
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to model requirements."""
        target_size = (self.config.max_image_size, self.config.max_image_size)
        
        # Use PIL for resizing
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        
        return np.array(pil_img)
    
    def _find_image_tokens(self, prompt: str) -> List[int]:
        """Find positions of image tokens in prompt."""
        # Look for special image token placeholders
        image_token = "<|image|>"
        positions = []
        start = 0
        
        while True:
            pos = prompt.find(image_token, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + len(image_token)
        
        return positions
    
    def _generate_image_token_ids(self, num_images: int) -> List[int]:
        """Generate token IDs for image tokens."""
        if not self.config.enable_multimodal:
            return []
        
        # Use vision processor to generate token IDs
        if hasattr(self.model_runner, 'vision_processor'):
            return self.model_runner.vision_processor.get_image_token_ids(num_images)
        
        # Fallback: simple sequential IDs
        base_id = 100000
        token_ids = []
        for img_idx in range(num_images):
            start_id = base_id + img_idx * 1000
            num_tokens = self.config.num_vision_tokens
            token_ids.extend(range(start_id, start_id + num_tokens))
        
        return token_ids
    
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        images: Optional[List[List[Union[str, np.ndarray, Image.Image]]]] = None,
        use_tqdm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate responses with multi-modal support.
        
        Args:
            prompts: List of text prompts or token IDs
            sampling_params: Sampling parameters
            images: Optional list of image lists for each prompt
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of generation results with text and metadata
        """
        if not self.config.enable_multimodal:
            # Fallback to standard generation
            return super().generate(prompts, sampling_params, use_tqdm)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        if images is None:
            images = [None] * len(prompts)
        
        for prompt, sp, img_list in zip(prompts, sampling_params, images):
            self.add_request(prompt, sp, img_list)
        
        # Generate responses
        outputs = {}
        while not self.is_finished():
            output, num_tokens = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # Process results
        results = []
        for seq_id in sorted(outputs):
            token_ids = outputs[seq_id]
            text = self.tokenizer.decode(token_ids)
            
            # Get sequence info
            seq = self.scheduler.running[0] if self.scheduler.running else None
            if seq and seq.seq_id == seq_id:
                result = {
                    "text": text,
                    "token_ids": token_ids,
                    "has_images": isinstance(seq, MultiModalSequence) and seq.has_images,
                    "num_vision_tokens": seq.num_vision_tokens if isinstance(seq, MultiModalSequence) else 0
                }
            else:
                result = {
                    "text": text,
                    "token_ids": token_ids,
                    "has_images": False,
                    "num_vision_tokens": 0
                }
            results.append(result)
        
        return results
    
    def get_multimodal_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-modal usage."""
        if not self.config.enable_multimodal:
            return {"multimodal_enabled": False}
        
        stats = {
            "multimodal_enabled": True,
            "vision_model": self.config.vision_model,
            "max_image_size": self.config.max_image_size,
            "num_vision_tokens": self.config.num_vision_tokens
        }
        
        if hasattr(self.model_runner, 'get_vision_stats'):
            stats.update(self.model_runner.get_vision_stats())
        
        return stats


class MultiModalLLM(MultiModalLLMEngine):
    """Convenience alias for MultiModalLLMEngine."""
    pass