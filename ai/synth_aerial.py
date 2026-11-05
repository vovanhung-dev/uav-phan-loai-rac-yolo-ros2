import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from common import CLASS_TO_ID, CLASSES, CUTOUT_DIR, DATASETS_DIR, TEXTURE_DIR, write_data_yaml, yolo_line

PIXEL_SIZE_RANGE = {
    'bottle': (34, 96),
    'plastic_bag': (44, 120),
    'battery': (16, 46),
    'organic_waste': (28, 84),
}


def load_cutouts(split):
    cutouts = {}
    for name in CLASSES:
        cutouts[name] = sorted((CUTOUT_DIR / split / name).glob('*.png'))
        if not cutouts[name]:
            raise SystemExit(f'Thiếu cutout {split}/{name}')
    return cutouts


def random_ground(textures, size, rng):
    texture = Image.open(rng.choice(textures)).convert('RGB')
    tile_scale = rng.uniform(0.35, 1.0)
    tile = texture.resize((max(64, int(texture.width * tile_scale)), max(64, int(texture.height * tile_scale))))
    padded = int(size * 1.5)
    canvas = Image.new('RGB', (padded, padded))
    for y in range(0, padded, tile.height):
        for x in range(0, padded, tile.width):
            canvas.paste(tile, (x, y))
    canvas = canvas.rotate(rng.uniform(0, 360), resample=Image.BILINEAR, expand=False)
    offset = (padded - size) // 2
    canvas = canvas.crop((offset, offset, offset + size, offset + size))
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.7, 1.25))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.75, 1.2))
    return canvas


def prepare_object(path, class_name, rng):
    obj = Image.open(path).convert('RGBA')
    if rng.random() < 0.5:
        obj = obj.transpose(Image.FLIP_LEFT_RIGHT)
    low, high = PIXEL_SIZE_RANGE[class_name]
    target = rng.uniform(low, high)
    scale = target / max(obj.size)
    obj = obj.resize((max(4, int(obj.width * scale)), max(4, int(obj.height * scale))), Image.LANCZOS)
    obj = obj.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
    rgb = obj.convert('RGB')
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.75, 1.2))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.8, 1.2))
    obj = Image.merge('RGBA', (*rgb.split(), obj.getchannel('A')))
    alpha = np.array(obj.getchannel('A'))
    ys, xs = np.where(alpha > 100)
    if len(xs) == 0:
        return None
    return obj.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def paste_with_shadow(canvas, obj, x, y, rng):
    alpha = obj.getchannel('A')
    offset = (int(rng.uniform(1, 5)), int(rng.uniform(1, 5)))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * rng.uniform(0.25, 0.45)))
    shadow = Image.new('RGBA', obj.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    canvas.paste(shadow, (x + offset[0], y + offset[1]), shadow)
    canvas.paste(obj, (x, y), obj)


def uav_shadow_layer(size, rng):
    span = rng.uniform(70, 190)
    layer = Image.new('L', (int(span * 1.4), int(span * 1.4)), 0)
    draw = ImageDraw.Draw(layer)
    center = layer.width / 2
    arm = span / 2
    rotor = span * rng.uniform(0.12, 0.2)
    thickness = max(3, int(span * rng.uniform(0.05, 0.09)))
    for angle in (45, 135, 225, 315):
        rad = np.deg2rad(angle)
        tip = (center + arm * np.cos(rad), center + arm * np.sin(rad))
        draw.line([(center, center), tip], fill=255, width=thickness)
        if rng.random() < 0.7:
            draw.ellipse([tip[0] - rotor, tip[1] - rotor, tip[0] + rotor, tip[1] + rotor], fill=255)
    body = span * rng.uniform(0.1, 0.16)
    draw.ellipse([center - body, center - body * 0.8, center + body, center + body * 0.8], fill=255)
    layer = layer.rotate(rng.uniform(0, 360), resample=Image.BICUBIC)
    layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.2)))
    strength = rng.uniform(0.3, 0.6)
    return layer.point(lambda v: int(v * strength))


def add_hard_negatives(canvas, rng):
    shadow_alpha = uav_shadow_layer(canvas.width, rng)
    shadow = Image.new('RGBA', shadow_alpha.size, (8, 10, 6, 0))
    shadow.putalpha(shadow_alpha)
    x = rng.randint(-shadow.width // 3, canvas.width - shadow.width * 2 // 3)
    y = rng.randint(-shadow.height // 3, canvas.height - shadow.height * 2 // 3)
    canvas.alpha_composite(shadow, (max(x, 0), max(y, 0)))


def compose_image(textures, cutouts, size, rng, max_objects, realistic=True, hard_negatives=False):
    canvas = random_ground(textures, size, rng).convert('RGBA')
    if hard_negatives and rng.random() < 0.5:
        add_hard_negatives(canvas, rng)
    labels = []
    placed = []
    if rng.random() < 0.05:
        object_count = 0
    else:
        object_count = rng.randint(1, max_objects)
    attempts = 0
    while len(placed) < object_count and attempts < object_count * 12:
        attempts += 1
        class_name = rng.choice(CLASSES)
        obj = prepare_object(rng.choice(cutouts[class_name]), class_name, rng)
        if obj is None or obj.width >= size or obj.height >= size:
            continue
        x = rng.randint(0, size - obj.width)
        y = rng.randint(0, size - obj.height)
        box = (x, y, x + obj.width, y + obj.height)
        if any(box_iou(box, other) > 0.08 for other in placed):
            continue
        if realistic:
            paste_with_shadow(canvas, obj, x, y, rng)
        else:
            canvas.paste(obj, (x, y), obj)
        placed.append(box)
        labels.append(yolo_line(CLASS_TO_ID[class_name], *box, size, size))
    image = canvas.convert('RGB')
    if realistic and rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    if realistic and rng.random() < 0.3:
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(2, 7), (size, size, 3))
        image = Image.fromarray(np.clip(np.array(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    return image, labels


def generate_split(out_dir, split, cutout_split, count, textures, size, rng, max_objects, realistic, hard_negatives):
    cutouts = load_cutouts(cutout_split)
    image_dir = out_dir / 'images' / split
    label_dir = out_dir / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        image, labels = compose_image(textures, cutouts, size, rng, max_objects, realistic, hard_negatives)
        stem = f'syn_{split}_{idx:05d}'
        image.save(image_dir / f'{stem}.jpg', quality=rng.randint(80, 95))
        (label_dir / f'{stem}.txt').write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
    print(split, count, 'ảnh')


def main():
    parser = argparse.ArgumentParser(description='Sinh ảnh rác tổng hợp góc nhìn UAV từ trên cao')
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--val', type=int, default=500)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--max-objects', type=int, default=9)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--name', default='synthetic')
    parser.add_argument('--plain', action='store_true', help='Tắt bóng đổ, nhiễu, mờ để làm thí nghiệm ablation')
    parser.add_argument('--hard-negatives', action='store_true', help='Chèn bóng UAV giả không gán nhãn làm mẫu âm khó')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    textures = sorted(TEXTURE_DIR.glob('*.jpg'))
    if not textures:
        raise SystemExit('Chưa có texture, chạy download_data.py trước')
    out_dir = DATASETS_DIR / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    generate_split(out_dir, 'train', 'train', args.train, textures, args.size, rng, args.max_objects, not args.plain, args.hard_negatives)
    generate_split(out_dir, 'val', 'val', args.val, textures, args.size, rng, args.max_objects, not args.plain, args.hard_negatives)
    print(write_data_yaml(out_dir, {'train': 'images/train', 'val': 'images/val'}))


if __name__ == '__main__':
    main()
