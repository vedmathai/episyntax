import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


class ACLPlotRenderer:
    SECTIONS = {
        'preliminary_results': {
            'title': 'Preliminary Results (R2-R4)',
            'order': [('R2', 'C4'), ('R2', 'C5'), ('R3', 'C1'), ('R3', 'C5'), ('R4', 'C3'), ('R4', 'C5')],
        },
        'main_results_presupposition': {
            'title': 'Presupposition (R5-R7)',
            'order': [('R5', 'E1'), ('R5', 'C1'), ('R6', 'E2'), ('R6', 'E1'), ('R7', 'E1'), ('R7', 'C3')],
        },
        'main_results_repair': {
            'title': 'Repair (R8-R10)',
            'order': [('R8', 'E3'), ('R8', 'C5'), ('R9', 'E4'), ('R9', 'C5'), ('R10', 'E5'), ('R10', 'C4')],
        },
        'main_results_presupposition_repair': {
            'title': 'Presupposition Repair (R11a-c)',
            'order': [('R11a', 'E6'), ('R11a', 'E3'), ('R11b', 'E7'), ('R11b', 'E4'), ('R11c', 'E8'), ('R11c', 'E5')],
        },
    }
    ALL_CONDITIONS = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'C1', 'C2', 'C3', 'C4', 'C5']

    COL_WIDTH = 3.3
    COLORS = {'GPT-5.5': '#4C72B0', 'gpt-oss-120b': '#DD8452', 'DeepSeek-V4': '#C44E52',
              'Llama-3.3': '#55A868', 'Human': '#333333'}
    MARKERS = {'GPT-5.5': 'o', 'gpt-oss-120b': 's', 'DeepSeek-V4': '^',
               'Llama-3.3': 'D', 'Human': '*'}
    OFFSETS = {'GPT-5.5': -0.28, 'gpt-oss-120b': -0.14, 'DeepSeek-V4': 0.0,
               'Llama-3.3': 0.14, 'Human': 0.28}
    ALPHA_LABEL_EXCLUDE = {'E5', 'C4'}

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._output_dir = self._repo_root / 'output' / 'figures'
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._all_data = None
        self._alpha_cache = None

        mpl.rcParams.update({
            'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
            'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'legend.fontsize': 6,
        })

    def load_data(self):
        with open(self._cache_dir / 'forest_ci_cache.pkl', 'rb') as f:
            all_data = pickle.load(f)
        with open(self._cache_dir / 'human_ci_cache.pkl', 'rb') as f:
            human_cache = pickle.load(f)
        with open(self._cache_dir / 'krippendorff_alpha_cache.pkl', 'rb') as f:
            self._alpha_cache = pickle.load(f)

        # Human data only exists for these 8 conditions; add NaN for the rest
        # so it behaves like every other model in the plotting loop below.
        human_full = {cond: human_cache.get(cond, (np.nan, np.nan, np.nan)) for cond in self.ALL_CONDITIONS}
        all_data['Human'] = human_full
        self._all_data = all_data
        return self._all_data

    def draw_row_shading(self, ax, rq_order):
        current_rq, shade = None, False
        for i, (rq, cond) in enumerate(rq_order):
            if rq != current_rq:
                current_rq = rq
                shade = not shade
            if shade:
                ax.axhspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.4, zorder=0)

        # Light connecting line between the two rows of each comparison pair
        # (distinct from the between-group shading above)
        n_rows = len(rq_order)
        for i in range(0, n_rows, 2):
            ax.axhline(i + 0.5, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.35, zorder=1)

    def draw_rq_labels(self, ax, rq_order):
        """R-label, shown once per pair, vertically centered on its two
        rows, placed to the left of the condition tick labels using a mixed
        axes/data transform so it stays at a fixed horizontal position
        regardless of the x-data range."""
        seen_rqs = []
        for i, (rq, cond) in enumerate(rq_order):
            if rq not in seen_rqs:
                seen_rqs.append(rq)
                pair_center = i + 0.5
                ax.text(-0.10, pair_center, rq, ha='right', va='center',
                        fontsize=7, fontweight='bold',
                        transform=ax.get_yaxis_transform(), clip_on=False)

    def draw_model_series(self, ax, section_key, rq_order, y_pos):
        for model_name, cache in self._all_data.items():
            # Human has no data for E6/E7/E8 (presupposition-repair
            # conditions aren't part of the human validation study) -- skip
            # it entirely for this section rather than show an empty legend
            # entry.
            if section_key == 'main_results_presupposition_repair' and model_name == 'Human':
                continue
            accs, los, his = [], [], []
            for rq, cond in rq_order:
                point, lo, hi = cache[cond]
                accs.append(point)
                los.append(lo)
                his.append(hi)
            accs, los, his = np.array(accs), np.array(los), np.array(his)
            ys = y_pos + self.OFFSETS[model_name]
            valid = ~np.isnan(accs)
            ax.errorbar(accs[valid], ys[valid],
                        xerr=[accs[valid] - los[valid], his[valid] - accs[valid]],
                        fmt=self.MARKERS[model_name], markersize=3.5, capsize=2,
                        color=self.COLORS[model_name], ecolor=self.COLORS[model_name],
                        elinewidth=0.9, alpha=0.9, label=model_name)

            # Krippendorff's alpha (human inter-annotator agreement for that
            # condition) printed immediately left of the Human CI whisker.
            if model_name == 'Human':
                for i, (rq, cond) in enumerate(rq_order):
                    if not valid[i] or cond not in self._alpha_cache or cond in self.ALPHA_LABEL_EXCLUDE:
                        continue
                    ax.text(los[i] - 0.03, ys[i], f'α={self._alpha_cache[cond]:.2f}',
                            ha='right', va='center', fontsize=5, style='italic',
                            color=self.COLORS['Human'], clip_on=False)

    def render_section(self, section_key, section):
        rq_order = section['order']
        n_rows = len(rq_order)
        fig, ax = plt.subplots(figsize=(self.COL_WIDTH, 0.42 * n_rows + 1.05))
        y_pos = np.arange(n_rows)
        # Y-tick labels show ONLY the condition name; the R-label is added
        # once per pair as a separate annotation to the left.
        labels = [cond for rq, cond in rq_order]

        self.draw_row_shading(ax, rq_order)
        self.draw_rq_labels(ax, rq_order)
        self.draw_model_series(ax, section_key, rq_order, y_pos)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Accuracy (95% CI)', labelpad=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.set_title(section['title'], fontsize=7.5, pad=3)
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        ax.tick_params(axis='both', length=2, pad=2)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16 + 0.010 * n_rows),
                  ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.35, fontsize=6)

        plt.tight_layout()
        plt.subplots_adjust(left=0.17)
        plt.savefig(self._output_dir / f'acl_col_{section_key}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self._output_dir / f'acl_col_{section_key}.pdf', bbox_inches='tight')
        print(f'Saved acl_col_{section_key}.png / .pdf')

    def render(self):
        self.load_data()
        for section_key, section in self.SECTIONS.items():
            self.render_section(section_key, section)


if __name__ == '__main__':
    renderer = ACLPlotRenderer()
    renderer.render()
