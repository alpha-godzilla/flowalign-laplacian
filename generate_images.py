import argparse
from pathlib import Path

import torch

from diffusion.editing.sd3_edit import get_editor
from piebench_utils import (
    load_image_tensor,
    load_json,
    resolve_image_path,
    resolve_piebench_layout,
    save_json,
    save_tensor_image,
)
from utils import util


class PIEBenchGenerator:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        editor_kwargs = {
            "model_key": args.model_key,
            "device": self.device,
            "shift": args.shift,
        }
        if args.method == "flowdinoalign":
            editor_kwargs.update(
                {
                    "dino_model": args.dino_model,
                    "dino_image_size": args.dino_image_size,
                    "dino_gamma": args.dino_gamma,
                    "dino_every_k": args.dino_every_k,
                }
            )
        if args.method == "flowlaplacianalign":
            editor_kwargs.update(
                {
                    "laplacian_image_size": args.laplacian_image_size,
                    "laplacian_levels": args.laplacian_levels,
                    "laplacian_gamma": args.laplacian_gamma,
                    "laplacian_every_k": args.laplacian_every_k,
                }
            )
        self.sampler = get_editor(args.method, **editor_kwargs).to(device=self.device)

    def load_records(self):
        dataset_path = Path(self.args.dataset_path)
        dataset_root, images_dir, edits_file = resolve_piebench_layout(dataset_path)
        records = load_json(edits_file)
        return dataset_root, images_dir, records

    def generate_one(self, record, dataset_root, images_dir):
        img_name = record.get("image", record.get("img"))
        src_prompt = record.get("source_prompt", record.get("src_prompt"))
        tgt_prompt = record.get("target_prompt", record.get("tgt_prompt"))
        sample_id = record.get("id", Path(img_name).stem)
        img_path = resolve_image_path(dataset_root, images_dir, img_name)

        src_img = load_image_tensor(img_path, img_size=(self.args.img_shape, self.args.img_shape))
        src_img = src_img * 2.0 - 1.0

        grad_methods = {"flowdinoalign", "flowlaplacianalign"}
        grad_context = torch.enable_grad() if self.args.method in grad_methods else torch.no_grad()
        with grad_context:
            pred = self.sampler.sample(
                src_img=src_img.to(self.device),
                src_prompt=src_prompt,
                tgt_prompt=tgt_prompt,
                null_prompt="",
                NFE=self.args.NFE,
                img_shape=(self.args.img_shape, self.args.img_shape),
                n_start=self.args.n_start,
                cfg_scale=self.args.cfg_scale,
            )

        out_path = self.output_dir / f"{sample_id}.png"
        save_tensor_image(pred, out_path)

        return {
            "id": sample_id,
            "image": img_name,
            "source_image": str(img_path),
            "pred_image": str(out_path),
            "source_prompt": src_prompt,
            "target_prompt": tgt_prompt,
            "mask": record.get("mask"),
        }

    def run(self):
        dataset_root, images_dir, records = self.load_records()
        if self.args.max_samples is not None:
            records = records[: self.args.max_samples]

        generated = []
        for idx, record in enumerate(records):
            print(f"生成图像 {idx + 1}/{len(records)}")
            try:
                generated.append(self.generate_one(record, dataset_root, images_dir))
                save_json(generated, self.output_dir / "prompts.json")
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")

        save_json(generated, self.output_dir / "prompts.json")
        print(f"✅ 生成完成，记录已保存到 {self.output_dir / 'prompts.json'}")
        return generated


def build_parser():
    parser = argparse.ArgumentParser(description="Generate PIE-Bench edited images with FlowAlign")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--model_key", type=str, default="/home/ljc/code/FlowAlign-main/stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors")
    parser.add_argument(
        "--method",
        type=str,
        default="flowalign",
        help="dual/sdedit/flowedit/flowalign/flowdinoalign/flowlaplacianalign",
    )
    parser.add_argument("--img_shape", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=13.5)
    parser.add_argument("--NFE", type=int, default=33)
    parser.add_argument("--n_start", type=int, default=17)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov2-small",
        help="DINOv2 model id or alias; aliases like dinov2_vits14 resolve to facebook/dinov2-small",
    )
    parser.add_argument("--dino_image_size", type=int, default=128)
    parser.add_argument("--dino_gamma", type=float, default=0.01)
    parser.add_argument("--dino_every_k", type=int, default=4)
    parser.add_argument("--laplacian_image_size", type=int, default=128)
    parser.add_argument("--laplacian_levels", type=int, default=2)
    parser.add_argument("--laplacian_gamma", type=float, default=0.01)
    parser.add_argument("--laplacian_every_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    util.set_seed(args.seed)
    PIEBenchGenerator(args).run()


if __name__ == "__main__":
    main()
