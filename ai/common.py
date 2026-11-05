import os
from pathlib import Path

CLASSES = ['bottle', 'plastic_bag', 'battery', 'organic_waste']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

DATA_ROOT = Path(os.environ.get('UAV_DATA_ROOT', Path.home() / 'uav_data'))
RAW_DIR = DATA_ROOT / 'raw'
CUTOUT_DIR = DATA_ROOT / 'cutouts'
TEXTURE_DIR = DATA_ROOT / 'textures'
DATASETS_DIR = DATA_ROOT / 'datasets'
RUNS_DIR = DATA_ROOT / 'runs'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'docs' / 'results'


def write_data_yaml(dataset_dir, splits):
    dataset_dir = Path(dataset_dir)
    lines = [f'path: {dataset_dir}']
    for split_name, relative in splits.items():
        lines.append(f'{split_name}: {relative}')
    lines.append(f'nc: {len(CLASSES)}')
    lines.append('names:')
    for idx, name in enumerate(CLASSES):
        lines.append(f'  {idx}: {name}')
    yaml_path = dataset_dir / 'data.yaml'
    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return yaml_path


def yolo_line(class_id, x1, y1, x2, y2, width, height):
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f'{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'
