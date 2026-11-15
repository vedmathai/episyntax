import csv
import html
import numpy as np
import pickle
from pathlib import Path


class HumanCIComputer:
    HUMAN_CONDITIONS = ['E1', 'C1', 'C3', 'C5', 'E3', 'E4', 'C4', 'E5']
    N_BOOT = 1000

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._cache_dir.mkdir(exist_ok=True)
        self._rng = np.random.default_rng(42)
        self._items = None

    def read_csv(self, path):
        with open(path, newline='') as f:
            return list(csv.DictReader(f))

    def merge_key(self, row):
        return html.unescape(row['Context']).strip() + '|' + html.unescape(row['What Lila said']).strip()

    def load_data(self):
        raw_rows = self.read_csv(self._repo_root / 'human_annotation' / 'data' / 'calibration_responses_anonymized.csv')
        tracking_rows = self.read_csv(self._repo_root / 'human_annotation' / 'analysis' / 'prolific_calibration_tracking.csv')
        tracking_by_key = {self.merge_key(row): row for row in tracking_rows}

        resp_cols = [c for c in raw_rows[0] if c.endswith('_Response')]
        excl_cols = [c for c in raw_rows[0] if c.endswith('_ExcludedFromAnalysis')]

        items = []
        for row in raw_rows:
            tracking_row = tracking_by_key.get(self.merge_key(row), {})
            items.append({
                'valid_responses': self.get_valid_responses(row, resp_cols, excl_cols),
                '_condition': tracking_row.get('_condition'),
                '_expected_answer': tracking_row.get('_expected_answer'),
            })
        self._items = items
        return self._items

    def get_valid_responses(self, row, resp_cols, excl_cols):
        return [float(row[rc]) for ec, rc in zip(excl_cols, resp_cols) if row[ec] != 'True']

    def score_lenient(self, score, expected):
        expected = str(expected).strip().lower()
        if 2.5 <= score <= 3 and expected == 'certain':
            return 1
        if -2.7 <= score <= 2.7 and expected == 'uncertain':
            return 1
        if -3 <= score <= -2.5 and expected == 'certainly not':
            return 1
        return 0

    def bootstrap_human_ci(self, items):
        boot_accs = np.empty(self.N_BOOT)
        for b in range(self.N_BOOT):
            correct_flags = []
            for item in items:
                vals = item['valid_responses']
                resampled = self._rng.choice(vals, size=len(vals), replace=True)
                correct_flags.append(self.score_lenient(np.mean(resampled), item['_expected_answer']))
            boot_accs[b] = np.mean(correct_flags)
        point = np.mean([self.score_lenient(np.mean(it['valid_responses']), it['_expected_answer']) for it in items])
        lo, hi = np.percentile(boot_accs, [2.5, 97.5])
        lo = min(lo, point)
        hi = max(hi, point)
        return point, lo, hi

    def compute(self):
        self.load_data()
        human_cache = {}
        for cond in self.HUMAN_CONDITIONS:
            sub = [it for it in self._items if it['_condition'] == cond]
            human_cache[cond] = self.bootstrap_human_ci(sub)
            print(f'{cond}: n={len(sub)}, point={human_cache[cond][0]:.3f}')

        with open(self._cache_dir / 'human_ci_cache.pkl', 'wb') as f:
            pickle.dump(human_cache, f)
        print('\nSaved human_ci_cache.pkl')
        return human_cache


if __name__ == '__main__':
    computer = HumanCIComputer()
    computer.compute()
