#!/usr/bin/env python3
"""
Validate a local diffusers-style model directory for Stable Diffusion 3 (local_files_only use-case).
Reports which components are present and, for any safetensors files, lists top keys to help identify component type.
"""
import argparse
from pathlib import Path

try:
    from safetensors import safe_open
except Exception:
    safe_open = None

REQUIRED_DIRS = [
    'unet',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'vae',
]

ROOT_FILES = [
    'model_index.json',
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json'
]

RELEASE_BUNDLE_FILES = [
    'sd3_medium.safetensors',
    'sd3_medium_incl_clips.safetensors',
    'sd3_medium_incl_clips_t5xxlfp8.safetensors',
    'sd3_medium_incl_clips_t5xxlfp16.safetensors',
]

RELEASE_TEXT_ENCODERS = [
    'text_encoders/clip_g.safetensors',
    'text_encoders/clip_l.safetensors',
    'text_encoders/t5xxl_fp8_e4m3fn.safetensors',
    'text_encoders/t5xxl_fp16.safetensors',
]


def inspect_safetensors(path: Path, max_keys=20):
    if safe_open is None:
        print(f"  - {path.name}: safetensors package not installed; skipping key inspection")
        return
    try:
        with safe_open(str(path), framework='pt', device='cpu') as f:
            keys = list(f.keys())
            print(f"  - {path.name}: {len(keys)} keys; sample keys:")
            for k in keys[:max_keys]:
                print(f"      {k}")
    except Exception as e:
        print(f"  - Failed to read {path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Local model directory to validate')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print('Model directory does not exist:', model_dir)
        return

    print('Checking root-level files:')
    for f in ROOT_FILES:
        p = model_dir / f
        print(' -', f, '→', 'FOUND' if p.exists() else 'MISSING')

    print('\nChecking expected component dirs:')
    for d in REQUIRED_DIRS:
        p = model_dir / d
        if p.exists() and any(p.iterdir()):
            print(f' - {d}: FOUND, contents:')
            for item in sorted(p.iterdir()):
                print('    ', item.name)
                if item.suffix in ['.safetensors', '.pt', '.bin']:
                    inspect_safetensors(item)
        else:
            print(f' - {d}: MISSING or empty')

    # Also list other safetensors in the root
    print('\nOther safetensors at root:')
    for item in sorted(model_dir.iterdir()):
        if item.suffix == '.safetensors' and item.is_file():
            print(' -', item.name)
            inspect_safetensors(item)

    print('\nChecking SD3 release bundle files:')
    for rel in RELEASE_BUNDLE_FILES:
        p = model_dir / rel
        print(' -', rel, '→', 'FOUND' if p.exists() else 'MISSING')

    print('\nChecking SD3 release text encoder files:')
    for rel in RELEASE_TEXT_ENCODERS:
        p = model_dir / rel
        print(' -', rel, '→', 'FOUND' if p.exists() else 'MISSING')

    print('\nValidation complete.')

if __name__ == '__main__':
    main()
