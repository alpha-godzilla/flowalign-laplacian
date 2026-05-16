from typing import List, Tuple
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_img_list(root: Path):
    if root.is_dir():
        files = list(sorted(root.glob('*.png'))) \
                + list(sorted(root.glob('*.jpg'))) \
                + list(sorted(root.glob('*.jpeg')))
    else:
        files = [root]

    for f in files:
        yield f

def load_img(img_path: Path, img_size:Tuple[int, int]=(1024, 1024)) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    img = img.resize(img_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor

def freq_decompose(latent, sigma, kernel_size):
    _, c, h, w = latent.shape

    factor = h//64
    k = kernel_size * factor + ((factor + 1) % 2)
    sigma = sigma * factor

    # decompose with a simple gaussian blur built from torch ops
    if isinstance(k, int):
        k = (k, k)
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))

    def gaussian_kernel_1d(kernel_size, std, device, dtype):
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(x ** 2) / (2 * std ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    sigma_x = max(float(sigma[0]), 1e-6)
    sigma_y = max(float(sigma[1]), 1e-6)
    kernel_x = gaussian_kernel_1d(k[1], sigma_x, latent.device, latent.dtype).view(1, 1, 1, -1)
    kernel_y = gaussian_kernel_1d(k[0], sigma_y, latent.device, latent.dtype).view(1, 1, -1, 1)

    pad_y = k[0] // 2
    pad_x = k[1] // 2
    x = F.pad(latent, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
    x = F.conv2d(x, kernel_x.expand(c, 1, 1, -1), groups=c)
    lp_latent = F.conv2d(x, kernel_y.expand(c, 1, -1, 1), groups=c)
    hp_latent = latent - lp_latent

    return lp_latent, hp_latent


def precompute_text_embedding(model, prompts: List[torch.Tensor], device: torch.device):
    for att in dir(model):
        if att.startswith('text_enc'):
            getattr(model, att).to(device)

    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            outputs.append(model.encode_prompt(prompt, batch_size=1))

    for att in dir(model):
        if att.startswith('text_enc'):
            delattr(model, att)

    torch.cuda.empty_cache()

    return outputs
