# 🖼️ Multi-Modal Inference Guide

This guide explains how to use the multi-modal inference capabilities in nano-vLLM, enabling image + text processing.

## 📋 Overview

The multi-modal feature adds vision capabilities to nano-vLLM, allowing you to:
- Process images alongside text prompts
- Use CLIP-based vision encoders
- Perform cross-modal attention between text and vision tokens
- Generate text descriptions for images
- Handle multiple images per prompt

## 🚀 Quick Start

### Basic Usage

```python
from nanovllm.engine.multimodal_llm_engine import MultiModalLLM
from nanovllm.sampling_params import SamplingParams
from PIL import Image

# Enable multi-modal inference
engine = MultiModalLLM(
    "qwen3-0.6b",
    enable_multimodal=True,
    vision_model="openai/clip-vit-base-patch32",
    max_image_size=224
)

# Load images
image = Image.open("cat.jpg")

# Generate text with image
prompts = ["Describe this image: <|image|>"]
params = SamplingParams(max_tokens=100)

results = engine.generate(prompts, params, images=[[image]])
print(results[0]["text"])
```

### Advanced Configuration

```python
# Full configuration options
engine = MultiModalLLM(
    "qwen3-0.6b",
    enable_multimodal=True,
    vision_model="openai/clip-vit-base-patch32",  # CLIP model
    max_image_size=224,                           # Resize images to 224x224
    num_vision_tokens=49,                         # 7x7 patch tokens per image
    freeze_vision_weights=True,                   # Keep vision encoder frozen
)
```

## 📸 Image Handling

### Supported Formats
- PIL Images (`PIL.Image`)
- NumPy arrays (`np.ndarray`)
- File paths (`str`)

### Image Preprocessing
Images are automatically:
- Resized to specified `max_image_size`
- Converted to RGB format
- Normalized for CLIP input

### Multiple Images
```python
# Multiple images per prompt
images = [Image.open("cat1.jpg"), Image.open("cat2.jpg")]
prompts = ["Compare these two images: <|image|> and <|image|>"]
```

## 🔧 Usage Patterns

### Text + Image Generation
```python
# Standard text + image
engine = MultiModalLLM("model-name", enable_multimodal=True)

# Single image
results = engine.generate(
    ["What's in this image? <|image|>"],
    SamplingParams(max_tokens=50),
    images=[[Image.open("photo.jpg")]]
)

# Multiple images
results = engine.generate(
    ["Describe both images: <|image|> <|image|>"],
    SamplingParams(max_tokens=100),
    images=[[Image.open("img1.jpg"), Image.open("img2.jpg")]]
)
```

### Vision Feature Extraction
```python
from nanovllm.layers.vision import VisionEncoder, VisionProcessor

# Direct vision processing
encoder = VisionEncoder(hidden_size=768)
processor = VisionProcessor(encoder)

# Extract features
images = [np.array(Image.open("image.jpg"))]
features, num_tokens = processor.process_images(images)
# features: [num_images * 49, hidden_size]
```

## ⚙️ Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_multimodal` | bool | `False` | Enable multi-modal features |
| `vision_model` | str | `"openai/clip-vit-base-patch32"` | Vision encoder model |
| `max_image_size` | int | `224` | Resize images to this size |
| `num_vision_tokens` | int | `49` | Number of vision tokens per image |
| `freeze_vision_weights` | bool | `True` | Keep vision model frozen |

## 🎯 Use Cases

### Image Captioning
```python
prompts = ["Generate a detailed caption for: <|image|>"]
```

### Visual Question Answering
```python
prompts = ["What color is the car in this image? <|image|>"]
```

### Document Understanding
```python
prompts = ["Extract text from this document: <|image|>"]
```

### Multi-Image Analysis
```python
prompts = ["Compare these two photos: <|image|> <|image|>"]
```

## 📊 Performance Considerations

### Memory Usage
- Vision encoder: ~400MB for CLIP base
- Each image: ~50KB for 49 tokens
- Batch processing: Memory scales linearly

### Speed Optimization
```python
# Use smaller vision model for faster inference
engine = MultiModalLLM(
    "qwen3-0.6b",
    enable_multimodal=True,
    vision_model="openai/clip-vit-small-patch16",
    max_image_size=224
)
```

## 🔍 Debugging

### Check Multi-modal Status
```python
stats = engine.get_multimodal_stats()
print(stats)
# Output: {'multimodal_enabled': True, 'vision_model': '...', ...}
```

### Vision Statistics
```python
if hasattr(engine.model_runner, 'get_vision_stats'):
    vision_stats = engine.model_runner.get_vision_stats()
    print(f"Vision cache size: {vision_stats['vision_cache_size']}")
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Images not processing
```python
# Check image format
print(f"Image shape: {image.size}, mode: {image.mode}")
# Ensure RGB format
if image.mode != 'RGB':
    image = image.convert('RGB')
```

**Issue**: High memory usage
```python
# Reduce image size
engine = MultiModalLLM(
    "model-name",
    enable_multimodal=True,
    max_image_size=128,  # Smaller size
    num_vision_tokens=16   # Fewer tokens
)
```

**Issue**: Placeholder tokens not working
```python
# Ensure correct placeholder syntax
prompt = "Your text <|image|> more text"
# Check processor
from nanovllm.layers.vision import VisionProcessor
processor = VisionProcessor(encoder)
modified, positions = processor.replace_image_placeholders(prompt, [image])
```

## 🔄 Backwards Compatibility

The multi-modal features are **opt-in** and **backwards compatible**:

```python
# Text-only (backwards compatible)
engine = MultiModalLLM("qwen3-0.6b")  # Works exactly like LLMEngine

# With multi-modal
engine = MultiModalLLM("qwen3-0.6b", enable_multimodal=True)
```

## 📚 API Reference

### MultiModalLLMEngine Methods

- `add_request(prompt, sampling_params, images=None)` - Add multi-modal request
- `generate(prompts, sampling_params, images=None, use_tqdm=True)` - Generate with images
- `get_multimodal_stats()` - Get multi-modal statistics

### VisionProcessor Methods

- `process_images(images)` - Extract vision features
- `replace_image_placeholders(text, images)` - Handle placeholders
- `get_image_token_ids(num_images)` - Generate token IDs

## 🚀 Next Steps

1. **Performance Tuning**: Experiment with different vision models and sizes
2. **Custom Vision Models**: Replace CLIP with domain-specific vision encoders
3. **Batch Processing**: Optimize for multiple images per batch
4. **Streaming**: Add support for streaming image processing