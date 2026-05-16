from pathlib import Path
import json
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def resolve_piebench_layout(dataset_path: Path):
    """Support both PIE_Bench_pp root and PIE_Bench_pp/preprocessed."""
    candidates = [dataset_path, dataset_path / "preprocessed"]
    for root in candidates:
        images_dir = root / "images"
        edits_file = root / "edits.json"
        if images_dir.exists() and edits_file.exists():
            return root, images_dir, edits_file
    raise FileNotFoundError(
        f"Could not find a usable dataset layout under {dataset_path}. "
        "Expected either <root>/edits.json + <root>/images or "
        "<root>/preprocessed/edits.json + <root>/preprocessed/images."
    )


def resolve_image_path(dataset_root: Path, images_dir: Path, img_name: str) -> Path:
    img_path = Path(str(img_name))
    candidates = [
        dataset_root / img_path,
        images_dir / img_path,
        images_dir / img_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_image_tensor(img_path: Path, img_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    if img_size is not None:
        img = img.resize(img_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_tensor_image(tensor: torch.Tensor, out_path: Path):
    img = tensor.detach().cpu()
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for saving, got shape {tuple(img.shape)}")
        img = img[0]
    if img.ndim != 3:
        raise ValueError(f"Expected 3D tensor for saving, got shape {tuple(img.shape)}")

    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    elif img.shape[-1] not in (1, 3):
        raise ValueError(f"Cannot infer channel layout from tensor shape {tuple(img.shape)}")

    img = img.float()
    if img.min().item() < 0:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    arr = (img.numpy() * 255).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def decode_pie_bench_mask(mask_str: str, size: int = 512) -> np.ndarray:
    """Decode PIE-Bench's flat 512x512 mask encoding."""
    num_pixels = size * size
    mask = np.zeros(num_pixels, dtype=np.uint8)
    if mask_str is None:
        return mask.reshape(size, size)

    tokens = [int(x) for x in str(mask_str).split() if x.strip()]
    if not tokens:
        return mask.reshape(size, size)

    if len(tokens) == 2 and tokens[0] == 0 and tokens[1] >= num_pixels:
        mask[:] = 1
        return mask.reshape(size, size)

    if len(tokens) % 2 == 0:
        for start, length in zip(tokens[0::2], tokens[1::2]):
            if length <= 0 or start >= num_pixels:
                continue
            end = min(num_pixels, start + length)
            if end > start:
                mask[start:end] = 1
        return mask.reshape(size, size)

    cursor = 0
    fill = 0
    for length in tokens:
        if length <= 0:
            fill = 1 - fill
            continue
        end = min(num_pixels, cursor + length)
        if fill == 1:
            mask[cursor:end] = 1
        cursor = end
        fill = 1 - fill
        if cursor >= num_pixels:
            break
    return mask.reshape(size, size)

