import csv
import numpy as np
import pickle
from pathlib import Path


class FinalCIComputer:
    CONDITION_TO_NUM = {'E1': 3, 'C1': 4, 'C3': 5, 'E3': 6, 'E4': 7, 'E5': 8, 'C4': 9, 'C5': 10,
                         'E2': 1, 'C2': 2, 'E6': 11, 'E7': 12, 'E8': 13}
    ALL_CONDITIONS = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'C1', 'C2', 'C3', 'C4', 'C5']
    N_BOOT = 1000

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._cache_dir.mkdir(exist_ok=True)
        self._task2_raw_dir = self._repo_root / 'results' / 'task2' / 'raw'
        self._model_files = {
            'GPT-5.5': self._task2_raw_dir / 'results_gpt5.csv',
            'gpt-oss-120b': self._task2_raw_dir / 'results_gpt_oss_120b.csv',
            'DeepSeek-V4': self._task2_raw_dir / 'results_deepseek.csv',
            'Llama-3.3': self._task2_raw_dir / 'results_llama70.csv',
        }
        self._answer_map = self.build_answer_map()
        self._rng = np.random.default_rng(42)

    def model_files(self):
        return self._model_files

    def read_csv(self, path):
        with open(path, newline='') as f:
            return list(csv.DictReader(f))

    def build_answer_map(self):
        answer_map = {}
        for cond in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
            for q in [1, 2]:
                answer_map[f'C1.{cond}_Q{q}'] = 'certain' if q == 1 else 'certainly not'
                answer_map[f'C2.{cond}_Q{q}'] = 'certainly not' if q == 1 else 'certain'
        for cond in [8, 9, 13]:
            for q in [1, 2]:
                answer_map[f'C1.{cond}_Q{q}'] = 'uncertain'
                answer_map[f'C2.{cond}_Q{q}'] = 'uncertain'
        return answer_map

    def lookup_expected(self, row):
        cond_num = self.CONDITION_TO_NUM[row['condition']]
        q_num = row['q_used'].replace('Q', '')
        return self._answer_map[f"{row['polarity_group']}.{cond_num}_Q{q_num}"]

    def to_score(self, raw_answer):
        try:
            return float(raw_answer)
        except (TypeError, ValueError):
            return float('nan')

    def score_lenient_vec(self, scores, expected):
        mean_scores = scores.mean(axis=1)
        certain_ok = (mean_scores >= 2.5) & (mean_scores <= 3) & (expected == 'certain')
        uncertain_ok = (mean_scores >= -2.7) & (mean_scores <= 2.7) & (expected == 'uncertain')
        certainly_not_ok = (mean_scores >= -3) & (mean_scores <= -2.5) & (expected == 'certainly not')
        return certain_ok | uncertain_ok | certainly_not_ok

    def item_groups(self, sub_rows):
        """Groups rows by (scenario_idx, condition, polarity_group, q_used,
        expected), keyed ascending the same way pandas' sorted groupby
        would, so the per-item ordering (and therefore the bootstrap RNG
        draw sequence below) is reproducible."""
        groups = {}
        for row in sub_rows:
            key = (int(row['scenario_idx']), row['condition'], row['polarity_group'],
                   row['q_used'], row['expected'])
            groups.setdefault(key, []).append(row['score'])
        return {key: groups[key] for key in sorted(groups)}

    def bootstrap_ci_fast(self, sub_rows):
        groups = self.item_groups(sub_rows)
        keys = list(groups)
        n_items = len(keys)
        max_templates = max(len(v) for v in groups.values())
        score_matrix = np.full((n_items, max_templates), np.nan)
        for i, key in enumerate(keys):
            vals = groups[key]
            score_matrix[i, :len(vals)] = vals
        expected_arr = np.array([key[4] for key in keys])

        point = self.score_lenient_vec(np.nanmean(score_matrix, axis=1, keepdims=True), expected_arr).mean()

        boot_accs = np.empty(self.N_BOOT)
        n_templates_per_item = (~np.isnan(score_matrix)).sum(axis=1)
        for b in range(self.N_BOOT):
            resampled_means = np.empty(n_items)
            for i in range(n_items):
                k = n_templates_per_item[i]
                idx = self._rng.integers(0, k, size=k)
                resampled_means[i] = score_matrix[i, :k][idx].mean()
            boot_accs[b] = self.score_lenient_vec(resampled_means.reshape(-1, 1), expected_arr).mean()

        lo, hi = np.percentile(boot_accs, [2.5, 97.5])
        lo = min(lo, point)
        hi = max(hi, point)
        return point, lo, hi

    def compute_model(self, path):
        rows = self.read_csv(path)
        for row in rows:
            row['score'] = self.to_score(row['raw_answer'])
            row['expected'] = self.lookup_expected(row)

        cache = {}
        for cond in self.ALL_CONDITIONS:
            sub = [row for row in rows if row['condition'] == cond]
            cache[cond] = self.bootstrap_ci_fast(sub) if len(sub) > 0 else (np.nan, np.nan, np.nan)
        n_scenarios = len({int(row['scenario_idx']) for row in rows})
        return cache, n_scenarios

    def compute(self):
        all_data = {}
        for model_name, path in self._model_files.items():
            cache, n_scenarios = self.compute_model(path)
            all_data[model_name] = cache
            print(f'{model_name} done ({n_scenarios} scenarios)')

        with open(self._cache_dir / 'forest_ci_cache.pkl', 'wb') as f:
            pickle.dump(all_data, f)
        print('Cached final bootstrap results to disk.')
        return all_data


if __name__ == '__main__':
    computer = FinalCIComputer()
    computer.compute()
