from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Args:
        temperature: Float that controls the randomness of the sampling.
        max_tokens: Maximum number of tokens to generate per output sequence.
        ignore_eos: Whether to ignore the End-Of-Sentence token.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
