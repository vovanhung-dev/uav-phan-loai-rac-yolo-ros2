import argparse
import io
import shutil

import pandas as pd
from PIL import Image

from common import CLASS_TO_ID, DATASETS_DIR, RAW_DIR, write_data_yaml, yolo_line
from make_cutouts import MOONDREAM_LABELS, moondream_split_of

SPLIT_TO_YOLO = {'train': 'train', 'val': 'val', 'sim': 'test'}


def export_split(frame, out_dir, rows, split):
    image_dir = out_dir / 'images' / split
    label_dir = out_dir / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in MOONDREAM_LABELS.values()}
    for row_idx in rows:
        row = frame.iloc[row_idx]
        image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        width, height = image.size
        lines = []
        for box, label in zip(row['boxes'], row['labels']):
            class_name = MOONDREAM_LABELS.get(label)
            if class_name is None:
                continue
            cx, cy, bw, bh = [float(v) for v in box]
            x1, x2 = max(0.0, (cx - bw / 2) * width), min(width, (cx + bw / 2) * width)
            y1, y2 = max(0.0, (cy - bh / 2) * height), min(height, (cy + bh / 2) * height)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(yolo_line(CLASS_TO_ID[class_name], x1, y1, x2, y2, width, height))
            counts[class_name] += 1
        stem = f'real_{row_idx:05d}'
        image.save(image_dir / f'{stem}.jpg', quality=95)
        (label_dir / f'{stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(split, len(rows), 'ảnh', counts)


def main():
    parser = argparse.ArgumentParser(description='Chuyển dữ liệu thật (chai, túi nilon) sang định dạng YOLO')
    parser.add_argument('--name', default='real')
    args = parser.parse_args()
    frame = pd.read_parquet(RAW_DIR / 'moondream' / 'data' / 'train-00000-of-00001.parquet')
    out_dir = DATASETS_DIR / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    buckets = {'train': [], 'val': [], 'test': []}
    for row_idx, labels in enumerate(frame['labels']):
        if any(label in MOONDREAM_LABELS for label in labels):
            buckets[SPLIT_TO_YOLO[moondream_split_of(row_idx)]].append(row_idx)
    for split, rows in buckets.items():
        export_split(frame, out_dir, rows, split)
    print(write_data_yaml(out_dir, {'train': 'images/train', 'val': 'images/val', 'test': 'images/test'}))


if __name__ == '__main__':
    main()
