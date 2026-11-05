import argparse
import zipfile

import requests
from huggingface_hub import snapshot_download

from common import RAW_DIR, TEXTURE_DIR

MOONDREAM_REPO = 'moondream/waste_detection'
GARBAGE_CLS_REPO = 'khoaliamle/Garbage_Classification_YOLO'
GARBAGE_CLS_FOLDERS = ['battery', 'biological', 'plastic', 'glass']
AMBIENTCG_TEXTURES = [
    'Grass001', 'Grass004', 'Grass005', 'Ground037', 'Ground054', 'Ground048',
    'Asphalt012', 'Gravel022', 'Concrete034', 'Ground080',
]


def download_moondream():
    target = RAW_DIR / 'moondream'
    snapshot_download(MOONDREAM_REPO, repo_type='dataset', local_dir=target, allow_patterns=['data/*.parquet'])
    print('moondream ->', target)


def download_garbage_classification():
    target = RAW_DIR / 'garbage_cls'
    patterns = [f'classify/*/{folder}/*' for folder in GARBAGE_CLS_FOLDERS]
    snapshot_download(GARBAGE_CLS_REPO, repo_type='dataset', local_dir=target, allow_patterns=patterns,
                      max_workers=16)
    print('garbage_cls ->', target)


def download_textures():
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in AMBIENTCG_TEXTURES:
        out_path = TEXTURE_DIR / f'{name}.jpg'
        if out_path.exists():
            continue
        zip_path = TEXTURE_DIR / f'{name}.zip'
        response = requests.get(f'https://ambientcg.com/get?file={name}_1K-JPG.zip', timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        with zipfile.ZipFile(zip_path) as archive:
            color_name = next(n for n in archive.namelist() if n.endswith('_Color.jpg'))
            out_path.write_bytes(archive.read(color_name))
        zip_path.unlink()
        print('texture ->', out_path)


def main():
    parser = argparse.ArgumentParser(description='Tải dữ liệu thô cho đồ án')
    parser.add_argument('--skip-moondream', action='store_true')
    parser.add_argument('--skip-classification', action='store_true')
    parser.add_argument('--skip-textures', action='store_true')
    args = parser.parse_args()
    if not args.skip_moondream:
        download_moondream()
    if not args.skip_classification:
        download_garbage_classification()
    if not args.skip_textures:
        download_textures()


if __name__ == '__main__':
    main()
