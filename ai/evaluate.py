import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

from common import CLASSES, DATASETS_DIR, RESULTS_DIR, RUNS_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COCO_BOTTLE_ID = 39
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
OPERATING_CONFIDENCE = 0.35

EVAL_SETS = {
    'syn_val': DATASETS_DIR / 'synthetic' / 'images' / 'val',
    'sim_test': DATASETS_DIR / 'gazebo_sim' / 'images' / 'test',
    'real_test': DATASETS_DIR / 'real' / 'images' / 'test',
    'sim_alt2': DATASETS_DIR / 'gazebo_alt2' / 'images' / 'test',
    'sim_alt4': DATASETS_DIR / 'gazebo_alt4' / 'images' / 'test',
}


def load_ground_truth(image_path):
    label_path = Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().split('\n'):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append([cls, (cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height])
    return image, np.array(boxes, dtype=np.float64).reshape(-1, 5)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


class ModelRunner:

    def __init__(self, weights, coco_baseline=False):
        self.model = YOLO(weights)
        self.coco_baseline = coco_baseline
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def predict(self, image, imgsz):
        result = self.model.predict(image, conf=0.001, iou=0.6, imgsz=imgsz, device=self.device, verbose=False,
                                    max_det=100)[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        if self.coco_baseline:
            keep = cls == COCO_BOTTLE_ID
            xyxy, conf, cls = xyxy[keep], conf[keep], np.zeros(keep.sum(), dtype=int)
        return np.column_stack([cls, conf, xyxy]) if len(cls) else np.zeros((0, 6))


def evaluate_set(runner, image_dir, imgsz, collect_examples=False):
    images = sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
    stats = {c: {'scores': [], 'tp': [], 'n_gt': 0} for c in range(len(CLASSES))}
    confusion = np.zeros((len(CLASSES) + 1, len(CLASSES) + 1), dtype=int)
    latencies = []
    examples = []
    for idx, image_path in enumerate(images):
        image, gt = load_ground_truth(image_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        pred = runner.predict(image, imgsz)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if idx >= 5:
            latencies.append((time.perf_counter() - started) * 1000)
        for c in range(len(CLASSES)):
            gt_c = gt[gt[:, 0] == c][:, 1:]
            pred_c = pred[pred[:, 0] == c]
            pred_c = pred_c[np.argsort(-pred_c[:, 1])]
            stats[c]['n_gt'] += len(gt_c)
            ious = iou_matrix(pred_c[:, 2:], gt_c)
            tp = np.zeros((len(pred_c), len(IOU_THRESHOLDS)), dtype=bool)
            for t_idx, threshold in enumerate(IOU_THRESHOLDS):
                matched = set()
                for p_idx in range(len(pred_c)):
                    if ious.shape[1] == 0:
                        break
                    candidates = [(ious[p_idx, g], g) for g in range(len(gt_c)) if g not in matched and ious[p_idx, g] >= threshold]
                    if candidates:
                        matched.add(max(candidates)[1])
                        tp[p_idx, t_idx] = True
            stats[c]['scores'].extend(pred_c[:, 1].tolist())
            stats[c]['tp'].extend(tp.tolist())
        operating = pred[pred[:, 1] >= OPERATING_CONFIDENCE]
        update_confusion(confusion, operating, gt)
        if collect_examples:
            examples.append((image_path, image, gt, operating))
    return summarize(stats, confusion, latencies), confusion, examples


def update_confusion(confusion, pred, gt):
    background = len(CLASSES)
    ious = iou_matrix(pred[:, 2:], gt[:, 1:])
    matched_pred = set()
    for g_idx in range(len(gt)):
        best_p, best_iou = None, 0.5
        for p_idx in range(len(pred)):
            if p_idx in matched_pred:
                continue
            if ious[p_idx, g_idx] >= best_iou:
                best_p, best_iou = p_idx, ious[p_idx, g_idx]
        if best_p is None:
            confusion[background, int(gt[g_idx, 0])] += 1
        else:
            matched_pred.add(best_p)
            confusion[int(pred[best_p, 0]), int(gt[g_idx, 0])] += 1
    for p_idx in range(len(pred)):
        if p_idx not in matched_pred:
            confusion[int(pred[p_idx, 0]), background] += 1


def summarize(stats, confusion, latencies):
    per_class = {}
    for c, name in enumerate(CLASSES):
        n_gt = stats[c]['n_gt']
        if n_gt == 0:
            continue
        scores = np.array(stats[c]['scores'])
        tp = np.array(stats[c]['tp'], dtype=bool).reshape(-1, len(IOU_THRESHOLDS))
        order = np.argsort(-scores)
        scores, tp = scores[order], tp[order]
        aps = []
        for t_idx in range(len(IOU_THRESHOLDS)):
            tp_cum = np.cumsum(tp[:, t_idx])
            fp_cum = np.cumsum(~tp[:, t_idx])
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            aps.append(average_precision(recall, precision) if len(scores) else 0.0)
        at_operating = scores >= OPERATING_CONFIDENCE
        tp_op = int(tp[at_operating, 0].sum())
        fp_op = int(at_operating.sum() - tp_op)
        precision_op = tp_op / max(tp_op + fp_op, 1)
        recall_op = tp_op / n_gt
        f1 = 2 * precision_op * recall_op / max(precision_op + recall_op, 1e-9)
        per_class[name] = {'n_gt': n_gt, 'AP50': aps[0], 'AP50_95': float(np.mean(aps)),
                           'precision': precision_op, 'recall': recall_op, 'F1': f1}
    summary = {
        'per_class': per_class,
        'mAP50': float(np.mean([v['AP50'] for v in per_class.values()])) if per_class else 0.0,
        'mAP50_95': float(np.mean([v['AP50_95'] for v in per_class.values()])) if per_class else 0.0,
        'macro_F1': float(np.mean([v['F1'] for v in per_class.values()])) if per_class else 0.0,
        'latency_ms': float(np.median(latencies)) if latencies else 0.0,
    }
    summary['fps'] = 1000.0 / summary['latency_ms'] if summary['latency_ms'] else 0.0
    return summary


def plot_confusion(confusion, title, path):
    labels = CLASSES + ['background']
    column_sums = confusion.sum(axis=0, keepdims=True)
    normalized = confusion / np.maximum(column_sums, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('Nhãn thật (ground truth)')
    ax.set_ylabel('Dự đoán')
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if confusion[i, j]:
                ax.text(j, i, f'{confusion[i, j]}\n{normalized[i, j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if normalized[i, j] > 0.5 else 'black')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_error_examples(examples, out_dir, limit=12):
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for image_path, image, gt, pred in examples:
        ious = iou_matrix(pred[:, 2:], gt[:, 1:])
        missed = sum(1 for g in range(len(gt)) if ious.shape[0] == 0 or ious[:, g].max() < 0.5)
        false_pos = sum(1 for p in range(len(pred)) if ious.shape[1] == 0 or ious[p].max() < 0.5)
        wrong_class = 0
        for p in range(len(pred)):
            if ious.shape[1] and ious[p].max() >= 0.5 and int(gt[ious[p].argmax(), 0]) != int(pred[p, 0]):
                wrong_class += 1
        errors = missed + false_pos + wrong_class
        if errors:
            scored.append((errors, image_path, image, gt, pred))
    scored.sort(key=lambda item: -item[0])
    for errors, image_path, image, gt, pred in scored[:limit]:
        canvas = image.copy()
        for cls, x1, y1, x2, y2 in gt:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(canvas, f'GT {CLASSES[int(cls)]}', (int(x1), int(y2) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for cls, conf, x1, y1, x2, y2 in pred:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(canvas, f'{CLASSES[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f'{image_path.stem}_err{errors}.jpg'), canvas)


def main():
    parser = argparse.ArgumentParser(description='Đánh giá các mô hình trên tập tổng hợp, mô phỏng Gazebo và ảnh thật')
    parser.add_argument('--experiments', nargs='+', default=None)
    parser.add_argument('--final', nargs='+', default=['E5_yolo11s_mix', 'E6_yolo11s_mix_hn'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--sets', nargs='+', default=list(EVAL_SETS))
    args = parser.parse_args()

    runners = {'B0_yolo11n_coco': ('yolo11n.pt', True)}
    experiment_names = args.experiments or sorted(p.parent.parent.name for p in RUNS_DIR.glob('*/weights/best.pt'))
    for name in experiment_names:
        runners[name] = (str(RUNS_DIR / name / 'weights' / 'best.pt'), False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / 'evaluation.json'
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else {}
    for name, (weights, coco) in runners.items():
        runner = ModelRunner(weights, coco)
        results.setdefault(name, {})
        for set_name in args.sets:
            image_dir = EVAL_SETS[set_name]
            if not image_dir.exists():
                print('bỏ qua', set_name, '(chưa có dữ liệu)')
                continue
            is_final = name in args.final
            summary, confusion, examples = evaluate_set(runner, image_dir, args.imgsz, collect_examples=is_final)
            results[name][set_name] = summary
            print(f"{name:20s} {set_name:10s} mAP50={summary['mAP50']:.3f} mAP50-95={summary['mAP50_95']:.3f} "
                  f"F1={summary['macro_F1']:.3f} {summary['latency_ms']:.1f}ms")
            if is_final:
                plot_confusion(confusion, f'{name} - {set_name}', RESULTS_DIR / f'confusion_{name}_{set_name}.png')
                save_error_examples(examples, RESULTS_DIR / 'errors' / name / set_name)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown_table(results)


def write_markdown_table(results):
    lines = ['| Mô hình | Tập | mAP@0.5 | mAP@0.5:0.95 | Macro-F1 | Độ trễ (ms) |', '|---|---|---|---|---|---|']
    for name, sets in results.items():
        for set_name, summary in sets.items():
            lines.append(f"| {name} | {set_name} | {summary['mAP50']:.3f} | {summary['mAP50_95']:.3f} | "
                         f"{summary['macro_F1']:.3f} | {summary['latency_ms']:.1f} |")
    lines += ['', '| Mô hình | Tập | Lớp | AP@0.5 | Precision | Recall | F1 | Số GT |', '|---|---|---|---|---|---|---|---|']
    for name, sets in results.items():
        for set_name, summary in sets.items():
            for cls, values in summary['per_class'].items():
                lines.append(f"| {name} | {set_name} | {cls} | {values['AP50']:.3f} | {values['precision']:.3f} | "
                             f"{values['recall']:.3f} | {values['F1']:.3f} | {values['n_gt']} |")
    (RESULTS_DIR / 'evaluation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
