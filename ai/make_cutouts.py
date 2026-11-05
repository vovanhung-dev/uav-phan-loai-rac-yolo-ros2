import argparse
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rembg import new_session, remove

from common import CUTOUT_DIR, RAW_DIR

MOONDREAM_LABELS = {'bottle': 'bottle', 'PlasticBag': 'plastic_bag'}
CLASSIFICATION_FOLDERS = {'battery': 'battery', 'biological': 'organic_waste'}
CLASSIFICATION_SPLIT_MAP = {'train': 'train', 'val': 'val', 'test': 'sim'}
MAX_SIDE = 320


def clean_alpha(rgba, min_coverage, max_coverage):
    alpha = np.array(rgba.getchannel('A'))
    mask = alpha > 128
    coverage = mask.mean()
    if coverage < min_coverage or coverage > max_coverage:
        return None
    ys, xs = np.where(mask)
    cropped = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = MAX_SIDE / max(cropped.size)
    if scale < 1:
        cropped = cropped.resize((max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))),
                                 Image.LANCZOS)
    return cropped


def save_cutout(rgba, split, class_name, stem):
    out_dir = CUTOUT_DIR / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba.save(out_dir / f'{stem}.png')


def moondream_split_of(row_idx):
    bucket = (row_idx * 2654435761) % 100
    if bucket < 75:
        return 'train'
    if bucket < 88:
        return 'val'
    return 'sim'


def moondream_crops(session, limits, rng):
    frame = pd.read_parquet(RAW_DIR / 'moondream' / 'data' / 'train-00000-of-00001.parquet')
    produced = {split: {name: 0 for name in MOONDREAM_LABELS.values()} for split in limits}
    order = list(range(len(frame)))
    rng.shuffle(order)
    for row_idx in order:
        row = frame.iloc[row_idx]
        labels = list(row['labels'])
        if not any(label in MOONDREAM_LABELS for label in labels):
            continue
        split = moondream_split_of(row_idx)
        image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        width, height = image.size
        for box_idx, (box, label) in enumerate(zip(row['boxes'], labels)):
            class_name = MOONDREAM_LABELS.get(label)
            if class_name is None or produced[split][class_name] >= limits[split]:
                continue
            cx, cy, bw, bh = [float(v) for v in box]
            x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
            y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue
            pad_x, pad_y = (x2 - x1) * 0.06, (y2 - y1) * 0.06
            crop = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))
            rgba = clean_alpha(remove(crop, session=session), 0.15, 0.97)
            if rgba is None:
                continue
            save_cutout(rgba, split, class_name, f'md_{row_idx}_{box_idx}')
            produced[split][class_name] += 1
    print('moondream', produced)


def classification_cutouts(session, limit_per_split, rng):
    root = RAW_DIR / 'garbage_cls' / 'classify'
    for source_split, target_split in CLASSIFICATION_SPLIT_MAP.items():
        for folder, class_name in CLASSIFICATION_FOLDERS.items():
            files = sorted((root / source_split / folder).glob('*'))
            rng.shuffle(files)
            produced = 0
            for path in files:
                if produced >= limit_per_split[target_split]:
                    break
                try:
                    image = Image.open(path).convert('RGB')
                except OSError:
                    continue
                rgba = clean_alpha(remove(image, session=session), 0.05, 0.92)
                if rgba is None:
                    continue
                save_cutout(rgba, target_split, class_name, f'cls_{source_split}_{path.stem}')
                produced += 1
            print(source_split, folder, '->', target_split, produced)


def main():
    parser = argparse.ArgumentParser(description='Tách nền vật thể rác thành cutout RGBA')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    session = new_session('u2net')

    moondream_crops(session, {'train': 380, 'val': 80, 'sim': 40}, rng)
    classification_cutouts(session, {'train': 380, 'val': 80, 'sim': 40}, rng)
    for split_dir in sorted(CUTOUT_DIR.iterdir()):
        for class_dir in sorted(split_dir.iterdir()):
            print(split_dir.name, class_dir.name, len(list(class_dir.glob('*.png'))))


if __name__ == '__main__':
    main()
