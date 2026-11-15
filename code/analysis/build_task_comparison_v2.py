import csv
import numpy as np
import pickle
from pathlib import Path


class TaskComparisonBuilder:
    SAMPLE_SCENARIOS = set(range(1, 11))
    N_BOOT = 1000
    CONDITION_TO_NUM = {'E1': 3, 'C1': 4, 'C3': 5, 'E3': 6, 'E4': 7, 'E5': 8, 'C4': 9, 'C5': 10,
                         'E2': 1, 'C2': 2, 'E6': 11, 'E7': 12, 'E8': 13}

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._cache_dir.mkdir(exist_ok=True)
        self._results_dir = self._repo_root / 'results'
        self._rng = np.random.default_rng(42)
        self._answer_map = self.build_answer_map()

        self._task2_files = {
            'GPT-5.5': self._results_dir / 'task2' / 'raw' / 'results_gpt5.csv',
            'gpt-oss-120b': self._results_dir / 'task2' / 'raw' / 'results_gpt_oss_120b.csv',
            'DeepSeek-V4': self._results_dir / 'task2' / 'raw' / 'results_deepseek.csv',
            'Llama-3.3': self._results_dir / 'task2' / 'raw' / 'results_llama70.csv',
        }
        self._task1_files = {
            'GPT-5.5': self._results_dir / 'task1' / 'raw' / 'task1_results_gpt5.csv',
            'gpt-oss-120b': self._results_dir / 'task1' / 'raw' / 'task1_results_gpt_oss_120b.csv',
            'DeepSeek-V4': self._results_dir / 'task1' / 'raw' / 'task1_results_deepseek.csv',
            'Llama-3.3': self._results_dir / 'task1' / 'raw' / 'task1_results_llama70.csv',
        }
        self._task3_files = {
            'Llama-3.3': self._results_dir / 'task3' / 'raw' / 'task3_results_llama70.csv',
            'DeepSeek-V4': self._results_dir / 'task3' / 'raw' / 'task3_results_deepseek.csv',
            'GPT-5.5': self._results_dir / 'task3' / 'raw' / 'task3_results_gpt5.csv',
            'gpt-oss-120b': self._results_dir / 'task3' / 'raw' / 'task3_results_gpt_oss_120b.csv',
        }

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

    def mode(self, values):
        """Smallest value among the most frequent, matching
        pandas.Series.mode().iloc[0]'s tie-break (ascending sort of ties)."""
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        return min(v for v, c in counts.items() if c == max_count)

    def item_groups(self, rows, item_cols):
        """Groups rows by item_cols, keyed ascending the same way pandas'
        sorted groupby would, so the per-item ordering (and therefore the
        bootstrap RNG draw sequence below) is reproducible. scenario_idx is
        coerced to int so it sorts numerically, not lexicographically."""
        groups = {}
        for row in rows:
            key = tuple(int(row[c]) if c == 'scenario_idx' else row[c] for c in item_cols)
            groups.setdefault(key, []).append(row)
        return {key: groups[key] for key in sorted(groups)}

    def bootstrap_task2_pooled(self, rows):
        item_cols = ['scenario_idx', 'condition', 'polarity_group', 'q_used', 'expected']
        groups = self.item_groups(rows, item_cols)
        keys = list(groups)
        n_items = len(keys)
        scores_per_item = [[r['score'] for r in groups[k]] for k in keys]
        max_t = max(len(v) for v in scores_per_item)
        score_matrix = np.full((n_items, max_t), np.nan)
        for i, vals in enumerate(scores_per_item):
            score_matrix[i, :len(vals)] = vals
        expected_arr = np.array([key[4] for key in keys])

        point = self.score_lenient_vec(np.nanmean(score_matrix, axis=1, keepdims=True), expected_arr).mean()
        n_per_item = (~np.isnan(score_matrix)).sum(axis=1)
        boot_accs = np.empty(self.N_BOOT)
        for b in range(self.N_BOOT):
            resampled_means = np.empty(n_items)
            for i in range(n_items):
                k = n_per_item[i]
                idx = self._rng.integers(0, k, size=k)
                resampled_means[i] = score_matrix[i, :k][idx].mean()
            boot_accs[b] = self.score_lenient_vec(resampled_means.reshape(-1, 1), expected_arr).mean()
        lo, hi = np.percentile(boot_accs, [2.5, 97.5])
        return point, min(lo, point), max(hi, point)

    def bootstrap_majority_pooled(self, rows, item_cols, expected_col):
        """Shared by Task 1 (labels) and Task 3 (binary choices): majority
        vote across the 5 templates per item, restricted to items with all
        5 templates present."""
        groups = self.item_groups(rows, item_cols)
        complete_keys = [key for key, group in groups.items() if len(group) == 5]
        n_items = len(complete_keys)
        expected_idx = item_cols.index(expected_col)
        label_matrix = [[r['raw_answer'] for r in groups[key]] for key in complete_keys]
        expected_arr = [key[expected_idx] for key in complete_keys]

        point = np.mean([self.mode(row) == expected_arr[i] for i, row in enumerate(label_matrix)])
        boot_accs = np.empty(self.N_BOOT)
        for b in range(self.N_BOOT):
            boot_correct = []
            for i in range(n_items):
                idx = self._rng.integers(0, 5, size=5)
                resampled_row = [label_matrix[i][j] for j in idx]
                boot_correct.append(self.mode(resampled_row) == expected_arr[i])
            boot_accs[b] = np.mean(boot_correct)
        lo, hi = np.percentile(boot_accs, [2.5, 97.5])
        return point, min(lo, point), max(hi, point), n_items

    def build_task2(self):
        task2_cache = {}
        for model, path in self._task2_files.items():
            rows = [row for row in self.read_csv(path) if int(row['scenario_idx']) in self.SAMPLE_SCENARIOS]
            for row in rows:
                row['score'] = self.to_score(row['raw_answer'])
                row['expected'] = self.lookup_expected(row)
            task2_cache[model] = self.bootstrap_task2_pooled(rows)
            print(f'Task 2 / {model}: point={task2_cache[model][0]:.3f}')
        return task2_cache

    def build_task1(self):
        item_cols = ['scenario_idx', 'condition', 'polarity_group', 'q_used', 'expected_answer']
        task1_cache = {}
        for model, path in self._task1_files.items():
            rows = self.read_csv(path)
            point, lo, hi, n = self.bootstrap_majority_pooled(rows, item_cols, 'expected_answer')
            task1_cache[model] = (point, lo, hi)
            print(f'Task 1 / {model}: {n} complete items, point={point:.3f}')
        return task1_cache

    def build_task3(self):
        item_cols = ['scenario_idx', 'pair', 'expected_choice']
        task3_cache = {}
        for model, path in self._task3_files.items():
            rows = self.read_csv(path)
            point, lo, hi, n = self.bootstrap_majority_pooled(rows, item_cols, 'expected_choice')
            task3_cache[model] = (point, lo, hi)
            # n reflects only items with all 5 templates present.
            print(f'Task 3 / {model}: {n} complete items, point={point:.3f}')
        return task3_cache

    def build(self):
        task2_cache = self.build_task2()
        task1_cache = self.build_task1()
        task3_cache = self.build_task3()

        with open(self._cache_dir / 'task_comparison_cache.pkl', 'wb') as f:
            pickle.dump({'Task 1': task1_cache, 'Task 2': task2_cache, 'Task 3': task3_cache}, f)
        print('\nSaved task_comparison_cache.pkl (before adding Human)')


if __name__ == '__main__':
    builder = TaskComparisonBuilder()
    builder.build()
