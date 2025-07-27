import torch
from typing import Optional, Tuple

class ThinkingParser:
    """解析思考模式的输出"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        
        # 获取特殊token的ID
        self.think_start_id = tokenizer.convert_tokens_to_ids(self.think_start_token)
        self.think_end_id = tokenizer.convert_tokens_to_ids(self.think_end_token)
    
    def parse_thinking_output(self, output_ids: torch.Tensor) -> Tuple[str, str]:
        """
        解析包含思考内容的输出
        返回: (thinking_content, final_content)
        """
        output_ids = output_ids.tolist()
        
        try:
            # 查找 </think> token
            think_end_idx = len(output_ids) - output_ids[::-1].index(self.think_end_id)
        except ValueError:
            # 没有找到结束标记
            think_end_idx = 0
        
        if think_end_idx > 0:
            thinking_content = self.tokenizer.decode(
                output_ids[:think_end_idx], 
                skip_special_tokens=True
            ).strip()
            final_content = self.tokenizer.decode(
                output_ids[think_end_idx:], 
                skip_special_tokens=True
            ).strip()
        else:
            thinking_content = ""
            final_content = self.tokenizer.decode(
                output_ids, 
                skip_special_tokens=True
            ).strip()
        
        return thinking_content, final_content 