import argparse
import json
import shutil
import time

from ultralytics import YOLO

from common import CLASSES, DATASETS_DIR, RESULTS_DIR, RUNS_DIR

EXPERIMENTS = {
    'E1_yolov8n_syn': {'weights': 'yolov8n.pt', 'train': ['synthetic'], 'note': 'Baseline: YOLOv8n, chỉ dữ liệu tổng hợp'},
    'E2_yolo11n_syn': {'weights': 'yolo11n.pt', 'train': ['synthetic'], 'note': 'YOLO11n, chỉ dữ liệu tổng hợp'},
    'E3_yolo11n_plain': {'weights': 'yolo11n.pt', 'train': ['synthetic_plain'], 'note': 'Ablation: tổng hợp không bóng đổ/nhiễu/mờ'},
    'E4_yolo11n_mix': {'weights': 'yolo11n.pt', 'train': ['synthetic', 'real'], 'note': 'YOLO11n, tổng hợp + ảnh thật'},
    'E5_yolo11s_mix': {'weights': 'yolo11s.pt', 'train': ['synthetic', 'real'], 'batch': 16, 'note': 'YOLO11s, tổng hợp + ảnh thật'},
    'E6_yolo11s_mix_hn': {'weights': 'yolo11s.pt', 'train': ['synthetic_hn', 'real'], 'batch': 16,
                          'note': 'Cải tiến: YOLO11s, tổng hợp có mẫu âm khó (bóng UAV) + ảnh thật'},
}


def build_data_yaml(experiment_name, train_sets):
    out_dir = DATASETS_DIR / 'mixes'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = [str(DATASETS_DIR / name / 'images' / 'train') for name in train_sets]
    val_paths = [str(DATASETS_DIR / 'synthetic' / 'images' / 'val')]
    if 'real' in train_sets:
        val_paths.append(str(DATASETS_DIR / 'real' / 'images' / 'val'))
    lines = ['train:'] + [f'  - {p}' for p in train_paths] + ['val:'] + [f'  - {p}' for p in val_paths]
    lines += [f'nc: {len(CLASSES)}', 'names:'] + [f'  {i}: {n}' for i, n in enumerate(CLASSES)]
    yaml_path = out_dir / f'{experiment_name}.yaml'
    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return yaml_path


def train_experiment(name, config, epochs, batch, imgsz, patience):
    data_yaml = build_data_yaml(name, config['train'])
    batch = config.get('batch', batch)
    model = YOLO(config['weights'])
    started = time.time()
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, patience=patience,
                project=str(RUNS_DIR), name=name, exist_ok=True, seed=0, deterministic=False,
                workers=8, cos_lr=True, close_mosaic=5, degrees=180, flipud=0.5, fliplr=0.5,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, scale=0.4, mosaic=1.0, verbose=False, plots=True)
    duration = time.time() - started
    best = RUNS_DIR / name / 'weights' / 'best.pt'
    record = {'experiment': name, 'note': config['note'], 'weights': config['weights'], 'train_sets': config['train'],
              'epochs': epochs, 'imgsz': imgsz, 'batch': batch, 'train_minutes': round(duration / 60, 1),
              'best': str(best)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / 'train_log.json'
    history = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else {}
    history[name] = record
    log_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    run_dir = RUNS_DIR / name
    plots_dir = RESULTS_DIR / 'training' / name
    plots_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ('results.png', 'results.csv', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png'):
        for path in run_dir.glob(pattern):
            shutil.copy(path, plots_dir / path.name)
    print(json.dumps(record, ensure_ascii=False))
    return best


def main():
    parser = argparse.ArgumentParser(description='Huấn luyện các thí nghiệm YOLO nhận diện rác')
    parser.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    for name in args.experiments:
        train_experiment(name, EXPERIMENTS[name], args.epochs, args.batch, args.imgsz, args.patience)


if __name__ == '__main__':
    main()
