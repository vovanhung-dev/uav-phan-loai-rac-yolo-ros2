import json

import matplotlib
import numpy as np

from common import RESULTS_DIR

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SET_LABELS = {'syn_val': 'Tổng hợp (val)', 'sim_test': 'Gazebo 3 m (test)', 'real_test': 'Ảnh thật (test)'}
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(results, metric, title, path):
    models = list(results)
    sets = [s for s in SET_LABELS if any(s in results[m] for m in models)]
    x = np.arange(len(models))
    width = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    style_axes(ax)
    for idx, set_name in enumerate(sets):
        values = [results[m].get(set_name, {}).get(metric, 0.0) for m in models]
        offset = (idx - (len(sets) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width - 0.02, color=SERIES_COLORS[idx], label=SET_LABELS[set_name],
                      edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xticks(x, [m.replace('_', '\n', 1) for m in models], color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, loc='left', fontsize=12)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -0.16), ncol=len(sets), labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def altitude_lines(results, path):
    models = [m for m in results if all(k in results[m] for k in ('sim_alt2', 'sim_test', 'sim_alt4'))]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style_axes(ax)
    altitudes = [2, 3, 4]
    highlight = [m for m in models if m.startswith(('E2', 'E5', 'E6'))][:3]
    for idx, model in enumerate(highlight):
        values = [results[model][k]['mAP50'] for k in ('sim_alt2', 'sim_test', 'sim_alt4')]
        ax.plot(altitudes, values, color=SERIES_COLORS[idx], linewidth=2, marker='o', markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=model)
    ax.set_xticks(altitudes, ['2 m', '3 m', '4 m'])
    ax.set_xlim(1.8, 4.2)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel('mAP@0.5', color=TEXT_SECONDARY)
    ax.set_title('Độ bền theo độ cao bay (Gazebo)', color=TEXT_PRIMARY, loc='left', pad=12)
    ax.legend(frameon=False, loc='lower left', labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    results = json.loads((RESULTS_DIR / 'evaluation.json').read_text(encoding='utf-8'))
    grouped_bars(results, 'mAP50', 'mAP@0.5 theo mô hình và tập đánh giá', RESULTS_DIR / 'chart_map50.png')
    grouped_bars(results, 'macro_F1', 'Macro-F1 (ngưỡng tin cậy 0.35)', RESULTS_DIR / 'chart_f1.png')
    altitude_lines(results, RESULTS_DIR / 'chart_altitude.png')
    print('Đã vẽ biểu đồ vào', RESULTS_DIR)


if __name__ == '__main__':
    main()
