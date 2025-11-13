import hashlib
import os
import urllib
import warnings
from typing import Union
import torch
from .clip_text import AnomalyCLIP

_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def _download(url: str, cache_dir: Union[str, None] = None):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2] if 'openaipublic' in url else ''
    download_target = os.path.join(cache_dir, filename)
    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
        else:
            return download_target
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")
    return download_target

def available_models():
    return list(_MODELS.keys())

def build_model(name: str, state_dict: dict, design_details=None):
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    model = AnomalyCLIP(embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details=design_details)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    return model.eval()

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details=None, download_root: str = None):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu")
    if state_dict is None:
        state_dict = model.state_dict()
    state_text = {k: v for k, v in state_dict.items() if not k.startswith("visual.")}
    model = build_model(name, state_text, design_details).to(device)
    if str(device) == "cpu":
        model.float()
    return model, None
