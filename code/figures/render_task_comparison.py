import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


class TaskComparisonRenderer:
    TASKS = ['Task 1', 'Task 2', 'Task 3']
    MODELS = ['GPT-5.5', 'gpt-oss-120b', 'DeepSeek-V4', 'Llama-3.3', 'Human']
    COLORS = {'GPT-5.5': '#4C72B0', 'gpt-oss-120b': '#DD8452', 'DeepSeek-V4': '#C44E52',
              'Llama-3.3': '#55A868', 'Human': '#333333'}
    MARKERS = {'GPT-5.5': 'o', 'gpt-oss-120b': 's', 'DeepSeek-V4': '^',
               'Llama-3.3': 'D', 'Human': '*'}
    OFFSETS = {'GPT-5.5': -0.28, 'gpt-oss-120b': -0.14, 'DeepSeek-V4': 0.0,
               'Llama-3.3': 0.14, 'Human': 0.28}
    COL_WIDTH = 3.3

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._output_dir = self._repo_root / 'output' / 'figures'
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._all_data = None

        mpl.rcParams.update({
            'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
            'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'legend.fontsize': 6,
        })

    def load_data(self):
        with open(self._cache_dir / 'task_comparison_cache.pkl', 'rb') as f:
            self._all_data = pickle.load(f)
        return self._all_data

    def draw_model_series(self, ax, y_pos):
        for model in self.MODELS:
            accs, los, his = [], [], []
            for task in self.TASKS:
                point, lo, hi = self._all_data[task][model]
                accs.append(point)
                los.append(lo)
                his.append(hi)
            accs, los, his = np.array(accs), np.array(los), np.array(his)
            ys = y_pos + self.OFFSETS[model]
            valid = ~np.isnan(accs)
            ax.errorbar(accs[valid], ys[valid],
                        xerr=[accs[valid] - los[valid], his[valid] - accs[valid]],
                        fmt=self.MARKERS[model], markersize=5, capsize=3,
                        color=self.COLORS[model], ecolor=self.COLORS[model],
                        elinewidth=1.1, alpha=0.9, label=model)

    def render(self):
        self.load_data()
        fig, ax = plt.subplots(figsize=(self.COL_WIDTH, 2.9))
        y_pos = np.arange(len(self.TASKS))

        for i in range(len(self.TASKS)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.4, zorder=0)

        self.draw_model_series(ax, y_pos)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.TASKS)
        ax.invert_yaxis()
        ax.set_xlabel('Accuracy (95% CI), pooled across all conditions')
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.set_title('Accuracy by Task, pooled across conditions', fontsize=6.5, pad=3)
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, frameon=False,
                  columnspacing=0.7, handletextpad=0.3, fontsize=6)

        plt.tight_layout()
        plt.savefig(self._output_dir / 'task_comparison_figure.png', dpi=300, bbox_inches='tight')
        plt.savefig(self._output_dir / 'task_comparison_figure.pdf', bbox_inches='tight')
        print('Saved task_comparison_figure.png / .pdf')


if __name__ == '__main__':
    renderer = TaskComparisonRenderer()
    renderer.render()
