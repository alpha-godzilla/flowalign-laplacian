#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from io import BytesIO
import base64
import pandas as pd
from PIL import Image


def extract_image_bytes(val):
    # 支持多种 parquet 存储 image 的格式
    if val is None:
        return None
    # case: dict with 'bytes'
    if isinstance(val, dict) and 'bytes' in val:
        b = val['bytes']
        # 如果是 list/tuple of ints
        if isinstance(b, (list, tuple)):
            return bytes(b)
        if isinstance(b, str):
            # 可能是 base64 字符串
            try:
                return base64.b64decode(b)
            except Exception:
                return b.encode('utf-8')
        return b
    # case: raw bytes
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    # case: base64 str
    if isinstance(val, str):
        try:
            return base64.b64decode(val)
        except Exception:
            # 不是 base64，可能是路径
            return None
    return None


def ensure_json(obj):
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return obj
    return obj


def main():
    parser = argparse.ArgumentParser(description='Preprocess PIE_Bench_pp parquet -> images + edits.json')
    parser.add_argument('--dataset_path', type=str, required=True, help='PIE_Bench_pp 根目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认写回 dataset_path/preprocessed）')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print('dataset_path 不存在:', dataset_path)
        return

    out_root = Path(args.output_dir) if args.output_dir else dataset_path / 'preprocessed'
    images_out = out_root / 'images'
    images_out.mkdir(parents=True, exist_ok=True)

    parquet_files = list(dataset_path.rglob('*.parquet'))
    if not parquet_files:
        print('未找到任何 parquet 文件')
        return

    edits = []
    saved = 0
    skipped = 0
    for pf in parquet_files:
        print('Reading', pf)
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print('  读取失败', pf, e)
            continue

        for i, row in df.iterrows():
            # 常见列映射
            img_val = None
            for cand in ['image', 'img', 'file', 'filename', 'file_name', 'image_name', 'image_id']:
                if cand in row.index:
                    img_val = row[cand]
                    break

            id_val = None
            for cand in ['id', 'image_id', 'img_id']:
                if cand in row.index:
                    id_val = row[cand]
                    break

            src_prompt = None
            for cand in ['source_prompt', 'src_prompt', 'source_text']:
                if cand in row.index:
                    src_prompt = row[cand]
                    break

            tgt_prompt = None
            for cand in ['target_prompt', 'tgt_prompt', 'target_text']:
                if cand in row.index:
                    tgt_prompt = row[cand]
                    break

            edit_action = row['edit_action'] if 'edit_action' in row.index else None
            aspect_mapping = row['aspect_mapping'] if 'aspect_mapping' in row.index else None
            blended_words = row['blended_words'] if 'blended_words' in row.index else None
            mask = row['mask'] if 'mask' in row.index else None

            # 处理图片
            img_bytes = extract_image_bytes(img_val)
            img_fname = None
            if img_bytes:
                try:
                    im = Image.open(BytesIO(img_bytes)).convert('RGB')
                    name = str(id_val) if id_val is not None else f"{pf.stem}_{i}"
                    img_fname = f"{name}.png"
                    im.save(images_out / img_fname, format='PNG')
                    saved += 1
                except Exception as e:
                    print('  保存图片失败:', e)
                    skipped += 1
                    img_fname = None
            else:
                # 如果 img_val 是路径字符串，尝试复制（不实现复制，直接记录路径）
                if isinstance(img_val, str) and img_val:
                    img_fname = img_val
                else:
                    skipped += 1

            entry = {
                'image': f"images/{img_fname}" if img_fname else '',
                'id': str(id_val) if id_val is not None else '',
                'source_prompt': str(src_prompt) if src_prompt is not None else '',
                'target_prompt': str(tgt_prompt) if tgt_prompt is not None else '',
                'edit_action': ensure_json(edit_action),
                'aspect_mapping': ensure_json(aspect_mapping),
                'blended_words': ensure_json(blended_words),
                'mask': ensure_json(mask)
            }
            edits.append(entry)

    # 保存 edits.json
    out_json = out_root / 'edits.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(edits, f, ensure_ascii=False, indent=2)

    print('\nFinished.')
    print('Saved images:', saved)
    print('Skipped rows (no image or failed):', skipped)
    print('edits.json saved to', out_json)

if __name__ == '__main__':
    main()
