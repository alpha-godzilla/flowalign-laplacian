import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import lpips
except Exception:
    lpips = None

try:
    import hpsv2
except Exception:
    hpsv2 = None

try:
    import open_clip
except Exception:
    open_clip = None

try:
    from torchmetrics.functional.image import peak_signal_noise_ratio as tm_psnr  # noqa: F401
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim  # noqa: F401
except Exception:
    tm_psnr = None
    tm_ssim = None


def _patch_hpsv2_vocab():
    """Workaround for hpsv2 bug where it misses the bpe vocab file in its own src/open_clip folder."""
    if hpsv2 is None or open_clip is None:
        return

    hps_dir = Path(hpsv2.__file__).parent
    target_file = hps_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    
    if not target_file.exists():
        clip_dir = Path(open_clip.__file__).parent
        source_file = clip_dir / "bpe_simple_vocab_16e6.txt.gz"
        
        if source_file.exists():
            print(f"  🔧 修复 HPSv2 Bug: 正在拷贝词表文件到 {target_file}")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_file, target_file)
        else:
            # Try to find it in the environment if direct import path fails
            pass

from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from piebench_utils import (
    decode_pie_bench_mask,
    load_image_tensor,
    load_json,
    resolve_image_path,
    resolve_piebench_layout,
    save_json,
)


@dataclass
class PIEBenchItem:
    sample_id: str
    source_image: Path
    pred_image: Path
    source_prompt: str
    target_prompt: str
    mask: Optional[str] = None


def _to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.float()
    if image.min().item() < 0:
        image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)
    arr = (image.numpy() * 255).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _resolve_dino_model_name(model_name: str) -> str:
    """Map human-friendly DINO aliases to real Hugging Face model ids."""
    aliases = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2-vits14": "facebook/dinov2-small",
        "facebook/dinov2_vits14": "facebook/dinov2-small",
        "facebook/dinov2-vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2-vitb14": "facebook/dinov2-base",
        "facebook/dinov2_vitb14": "facebook/dinov2-base",
        "facebook/dinov2-vitb14": "facebook/dinov2-base",
    }
    return aliases.get(model_name, model_name)


class PIEBenchEvaluator:
    def __init__(
        self,
        dataset_path: str,
        pred_dir: str,
        output_json: str,
        device: str = "cuda",
        source_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        prompts_json: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        dino_model_name: str = "facebook/dinov2-small",
        lpips_net: str = "vgg",
        hps_version: str = "v2.1",
        mask_base_size: int = 512,
    ):
        self.dataset_path = Path(dataset_path)
        self.pred_dir = Path(pred_dir)
        self.output_json = Path(output_json)
        self.device = torch.device(device)
        self.source_dir = Path(source_dir) if source_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.prompts_json = Path(prompts_json) if prompts_json else self.pred_dir / "prompts.json"
        self.clip_model_name = clip_model_name
        self.dino_model_name = dino_model_name
        self.lpips_net = lpips_net
        self.hps_version = hps_version
        self.mask_base_size = mask_base_size

        self.dataset_root, self.dataset_images_dir, self.edits_file = resolve_piebench_layout(self.dataset_path)
        self.records = self._load_records()
        
        # Apply patch for HPSv2 bug before initializing models
        _patch_hpsv2_vocab()
        
        self.lpips_model = self._init_lpips()
        self.clip_processor, self.clip_model = self._init_clip()
        self.dino_processor, self.dino_model = self._init_dino()

        self.results: Dict[str, Any] = {
            "samples": [],
            "metrics": {
                "background_mse": [],
                "background_psnr": [],
                "background_ssim": [],
                "background_lpips_vgg": [],
                "structural_distance_dino": [],
                "clip_score": [],
                "hps_score": [],
            },
        }

    def _init_lpips(self):
        if lpips is None:
            raise RuntimeError("lpips is required for background LPIPS. Please install a compatible lpips stack.")
        return lpips.LPIPS(net=self.lpips_net).to(self.device).eval()

    def _init_clip(self):
        processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
        model.requires_grad_(False)
        return processor, model

    def _init_dino(self):
        model_name = _resolve_dino_model_name(self.dino_model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        model.requires_grad_(False)
        return processor, model

    def _load_records(self) -> List[PIEBenchItem]:
        def resolve_source_path(raw_path: str, image_fallback: Optional[str] = None) -> Path:
            candidate = Path(raw_path)
            if candidate.is_absolute() and candidate.exists():
                return candidate
            if self.source_dir is not None:
                for base in [self.source_dir, self.dataset_images_dir, self.dataset_root]:
                    test_path = base / candidate
                    if test_path.exists():
                        return test_path
                if image_fallback is not None:
                    fallback = self.source_dir / Path(image_fallback).name
                    if fallback.exists():
                        return fallback
            for base in [self.dataset_images_dir, self.dataset_root]:
                test_path = base / candidate
                if test_path.exists():
                    return test_path
            return candidate

        def resolve_pred_path(raw_path: str, sample_id: str) -> Path:
            candidate = Path(raw_path)
            if candidate.is_absolute() and candidate.exists():
                return candidate
            
            # Candidates for filenames: exact name, sample_id.png, or sample_id_edited.png
            names = [raw_path]
            if not raw_path.endswith(f"{sample_id}.png"):
                names.append(f"{sample_id}.png")
            names.append(f"{sample_id}_edited.png")
            
            # Search in the provided directory and common subdirectories
            for base in [self.pred_dir, self.pred_dir / "edited", self.pred_dir / "predictions"]:
                for name in names:
                    test_path = base / name
                    if test_path.exists():
                        return test_path
            
            return candidate

        if self.prompts_json.exists():
            payload = load_json(self.prompts_json)
            records = []
            for row in payload:
                sample_id = str(row.get("id") or Path(row.get("image", "")).stem)
                source_image = resolve_source_path(row.get("source_image", row.get("image", sample_id)), row.get("image"))
                pred_image = resolve_pred_path(row.get("pred_image", f"{sample_id}.png"), sample_id)
                records.append(
                    PIEBenchItem(
                        sample_id=sample_id,
                        source_image=source_image,
                        pred_image=pred_image,
                        source_prompt=row.get("source_prompt", ""),
                        target_prompt=row.get("target_prompt", ""),
                        mask=row.get("mask"),
                    )
                )
            return records

        _, images_dir, edits_file = resolve_piebench_layout(self.dataset_path)
        edits = load_json(edits_file)
        records = []
        for row in edits:
            img_name = row.get("image", row.get("img"))
            sample_id = str(row.get("id") or Path(img_name).stem)
            source_image = resolve_image_path(self.dataset_root, images_dir, img_name)
            pred_image = resolve_pred_path(f"{sample_id}.png", sample_id)
            records.append(
                PIEBenchItem(
                    sample_id=sample_id,
                    source_image=source_image,
                    pred_image=pred_image,
                    source_prompt=row.get("source_prompt", row.get("src_prompt", "")),
                    target_prompt=row.get("target_prompt", row.get("tgt_prompt", "")),
                    mask=row.get("mask"),
                )
            )
        return records

    def _load_mask(self, item: PIEBenchItem, image_size: int) -> torch.Tensor:
        if self.mask_dir is not None:
            candidates = [
                self.mask_dir / f"{item.sample_id}.png",
                self.mask_dir / f"{item.sample_id}.jpg",
                self.mask_dir / f"{Path(item.source_image).stem}.png",
                self.mask_dir / f"{Path(item.source_image).stem}.jpg",
            ]
            for candidate in candidates:
                if candidate.exists():
                    mask_img = Image.open(candidate).convert("L")
                    mask_arr = (np.asarray(mask_img, dtype=np.float32) > 127).astype(np.float32)
                    mask_t = torch.from_numpy(mask_arr)[None, None]
                    if mask_t.shape[-2:] != (image_size, image_size):
                        mask_t = F.interpolate(mask_t, size=(image_size, image_size), mode="nearest")
                    return mask_t.to(self.device)

        mask_arr = decode_pie_bench_mask(item.mask, size=self.mask_base_size).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr)[None, None]
        if mask_t.shape[-2:] != (image_size, image_size):
            mask_t = F.interpolate(mask_t, size=(image_size, image_size), mode="nearest")
        return mask_t.to(self.device)

    def _compute_masked_mse(self, src: torch.Tensor, pred: torch.Tensor, bg_mask: torch.Tensor) -> float:
        diff2 = (src - pred) ** 2
        denom = bg_mask.sum() * src.shape[1]
        if denom.item() <= 0:
            return float("nan")
        return float((diff2 * bg_mask).sum().item() / denom.item())

    def _compute_masked_psnr(self, src: torch.Tensor, pred: torch.Tensor, bg_mask: torch.Tensor) -> float:
        mse = self._compute_masked_mse(src, pred, bg_mask)
        if not math.isfinite(mse):
            return float("nan")
        if mse <= 0:
            return float("inf")
        return float(10.0 * math.log10(1.0 / mse))

    def _compute_masked_ssim(self, src: torch.Tensor, pred: torch.Tensor, bg_mask: torch.Tensor) -> float:
        # src/pred: [B, C, H, W], bg_mask: [B, 1, H, W]
        if src.ndim != 4 or pred.ndim != 4 or bg_mask.ndim != 4:
            raise ValueError("SSIM expects [B,C,H,W] tensors and [B,1,H,W] mask.")
        b, c, h, w = src.shape
        window_size = min(11, h, w)
        if window_size % 2 == 0:
            window_size = max(3, window_size - 1)
        if window_size < 3:
            return float("nan")

        window = _gaussian_window(window_size, 1.5, c, src.device, src.dtype)
        pad = window_size // 2

        mask_c = bg_mask.expand(-1, c, -1, -1)
        den = F.conv2d(bg_mask, window[:1], padding=pad).clamp_min(1e-6)
        den_c = den.expand(-1, c, -1, -1)

        mu_x = F.conv2d(src * mask_c, window, padding=pad, groups=c) / den_c
        mu_y = F.conv2d(pred * mask_c, window, padding=pad, groups=c) / den_c
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(src * src * mask_c, window, padding=pad, groups=c) / den_c - mu_x2
        sigma_y2 = F.conv2d(pred * pred * mask_c, window, padding=pad, groups=c) / den_c - mu_y2
        sigma_xy = F.conv2d(src * pred * mask_c, window, padding=pad, groups=c) / den_c - mu_xy

        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        valid = (den > 1e-6).expand_as(ssim_map)
        if not valid.any():
            return float("nan")
        return float(ssim_map[valid].mean().item())

    def _compute_background_lpips(self, src: torch.Tensor, pred: torch.Tensor, bg_mask: torch.Tensor) -> float:
        src_bg = src * bg_mask
        pred_bg = pred * bg_mask
        with torch.no_grad():
            score = self.lpips_model(src_bg * 2 - 1, pred_bg * 2 - 1).item()
        return float(score)

    def _compute_structural_distance(self, src: torch.Tensor, pred: torch.Tensor, bg_mask: torch.Tensor) -> float:
        src_bg = (src * bg_mask).clamp(0.0, 1.0)
        pred_bg = (pred * bg_mask).clamp(0.0, 1.0)
        src_pil = _to_pil(src_bg)
        pred_pil = _to_pil(pred_bg)
        inputs = self.dino_processor(images=[src_pil, pred_pil], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dino_model(pixel_values=inputs["pixel_values"])
            feat = out.last_hidden_state.mean(dim=1)
            feat = F.normalize(feat, dim=1)
        return float((1.0 - F.cosine_similarity(feat[0:1], feat[1:2]).item()))

    def _compute_clip_score(self, pred: torch.Tensor, target_prompt: str) -> float:
        pred_pil = _to_pil(pred.clamp(0.0, 1.0))
        inputs = self.clip_processor(text=[target_prompt], images=[pred_pil], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            image_emb = F.normalize(out.image_embeds, dim=-1)
            text_emb = F.normalize(out.text_embeds, dim=-1)
            score = (image_emb * text_emb).sum(dim=-1).item()
        return float(score)

    def _compute_hps_score(self, pred: torch.Tensor, target_prompt: str) -> float:
        if hpsv2 is None:
            raise NotImplementedError(
                "hpsv2 library is not installed. Please install it with 'pip install hpsv2'."
            )
        pred_pil = _to_pil(pred.clamp(0.0, 1.0))
        with torch.no_grad():
            # hpsv2.score returns a list of scores for a list of images
            scores = hpsv2.score([pred_pil], target_prompt, hps_version=self.hps_version)
        return float(scores[0])

    def _save_results(self):
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.results, self.output_json)

    def _summarize(self) -> Dict[str, Any]:
        summary = {}
        for metric_name, values in self.results["metrics"].items():
            clean = [v for v in values if v is not None and math.isfinite(float(v))]
            if clean:
                summary[metric_name] = {
                    "mean": float(np.mean(clean)),
                    "std": float(np.std(clean)),
                    "count": int(len(clean)),
                }
            else:
                summary[metric_name] = {"mean": None, "std": None, "count": 0}
        self.results["summary"] = summary
        return summary

    def evaluate(self):
        for idx, item in enumerate(self.records):
            print(f"评估图像 {idx + 1}/{len(self.records)}")
            if not item.pred_image.exists():
                print(f"  ❌ 预测图像不存在: {item.pred_image}")
                continue

            src = load_image_tensor(item.source_image)
            pred = load_image_tensor(item.pred_image, img_size=src.shape[-2:])
            if pred.shape[-2:] != src.shape[-2:]:
                pred = F.interpolate(pred, size=src.shape[-2:], mode="bilinear", align_corners=False)

            bg_mask = self._load_mask(item, image_size=src.shape[-1])
            if bg_mask.shape[-2:] != src.shape[-2:]:
                bg_mask = F.interpolate(bg_mask, size=src.shape[-2:], mode="nearest")
            bg_mask = bg_mask.clamp(0.0, 1.0)
            bg_mask = (1.0 - bg_mask).to(self.device)  # 1 means background

            src = src.to(self.device)
            pred = pred.to(self.device)

            bg_mse = self._compute_masked_mse(src, pred, bg_mask)
            bg_psnr = self._compute_masked_psnr(src, pred, bg_mask)
            bg_ssim = self._compute_masked_ssim(src, pred, bg_mask)
            bg_lpips = self._compute_background_lpips(src, pred, bg_mask)
            dino_dist = self._compute_structural_distance(src, pred, bg_mask)
            clip_score = self._compute_clip_score(pred, item.target_prompt)

            try:
                hps_score = self._compute_hps_score(pred, item.target_prompt)
            except Exception as e:
                print(f"  ⚠️  HPS 错误: {e}")
                hps_score = None

            self.results["samples"].append(
                {
                    "id": item.sample_id,
                    "source_image": str(item.source_image),
                    "pred_image": str(item.pred_image),
                    "source_prompt": item.source_prompt,
                    "target_prompt": item.target_prompt,
                    "background_mse": bg_mse,
                    "background_psnr": bg_psnr,
                    "background_ssim": bg_ssim,
                    "background_lpips_vgg": bg_lpips,
                    "structural_distance_dino": dino_dist,
                    "clip_score": clip_score,
                    "hps_score": hps_score,
                }
            )
            self.results["metrics"]["background_mse"].append(bg_mse)
            self.results["metrics"]["background_psnr"].append(bg_psnr)
            self.results["metrics"]["background_ssim"].append(bg_ssim)
            self.results["metrics"]["background_lpips_vgg"].append(bg_lpips)
            self.results["metrics"]["structural_distance_dino"].append(dino_dist)
            self.results["metrics"]["clip_score"].append(clip_score)
            if hps_score is not None:
                self.results["metrics"]["hps_score"].append(hps_score)

            print(
                "  ✓ "
                f"bg_mse={bg_mse:.6f}, bg_psnr={bg_psnr:.4f}, bg_ssim={bg_ssim:.4f}, "
                f"bg_lpips_vgg={bg_lpips:.4f}, dino={dino_dist:.4f}, clip={clip_score:.4f}, "
                f"hps={hps_score if hps_score is not None else 'N/A'}"
            )
            self._save_results()

        summary = self._summarize()
        self._save_results()
        final_stats = {
            "background_mse": summary["background_mse"]["mean"],
            "background_psnr": summary["background_psnr"]["mean"],
            "background_ssim": summary["background_ssim"]["mean"],
            "background_lpips_vgg": summary["background_lpips_vgg"]["mean"],
            "structural_distance_dino": summary["structural_distance_dino"]["mean"],
            "clip_score": summary["clip_score"]["mean"],
            "hps_score": summary["hps_score"]["mean"],
        }
        print("\nFinal statistics:")
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))
        return final_stats


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate PIE-Bench metrics from local generated images")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, default="eval_results")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--prompts_json", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov2-small",
        help="DINOv2 model id or alias; aliases like dinov2_vits14 resolve to facebook/dinov2-small",
    )
    parser.add_argument("--lpips_net", type=str, default="vgg")
    parser.add_argument("--hps_version", type=str, default="v2.1")
    parser.add_argument("--mask_base_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    pred_dir = Path(args.pred_dir)
    output_json = Path(args.output_json) if args.output_json else pred_dir / "metrics.json"
    prompts_json = args.prompts_json if args.prompts_json else str(pred_dir / "prompts.json")
    evaluator = PIEBenchEvaluator(
        dataset_path=args.dataset_path,
        pred_dir=str(pred_dir),
        output_json=str(output_json),
        device=args.device,
        source_dir=args.source_dir,
        mask_dir=args.mask_dir,
        prompts_json=prompts_json,
        clip_model_name=args.clip_model,
        dino_model_name=args.dino_model,
        lpips_net=args.lpips_net,
        hps_version=args.hps_version,
        mask_base_size=args.mask_base_size,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
