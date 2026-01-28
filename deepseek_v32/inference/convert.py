import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_a_proj": ("wq_a", None),
    # 不进行切分，每个gpu都复制一份
    "q_a_layernorm": ("q_norm", None),
    # 按照第0维进行切分，通常是column parallel，切分输出维度
    # wq_b是对输出维度中的head进行切分，分配到不同gpu上计算
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    # 沿着第1维进行切分，通常是row parallel，切分输入维度
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
    "wq_b": ("wq_b", None),
    "wk": ("wk", None),
    "k_norm": ("k_norm", None),
    "weights_proj": ("weights_proj", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    # 根据mp(model parallelism)参数，提前对权重进行切分，保存为model0-mp8.safetensors,
    # model1-mp8.safetensors, ..., model7-mp8.safetensors，这样gpu0加载model0、gpu1
    # 加载model1
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        # e.g., layers.0.experts.5.w1.weight
                        idx = int(name.split(".")[-3])
                        # 必须在[i * n_local_experts, (i + 1) * n_local_experts)范围之内
                        # 属于第i个gpu负责的experts
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    # 根据mapping设定的哪个维度(输入还是输出)进行划分
                    elif dim is not None:
                        assert param.size(
                            dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        # 对param第dim维进行切分操作，只保留第i个shard
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
