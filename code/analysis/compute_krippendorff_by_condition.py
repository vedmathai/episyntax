import csv
import html
import pickle
from pathlib import Path

from iaa_metrics import krippendorff_alpha_ragged, gwet_ac1_ragged


class KrippendorffByConditionComputer:
    ALL_CONDITIONS = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'C1', 'C2', 'C3', 'C4', 'C5']

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._cache_dir = self._repo_root / 'cache'
        self._cache_dir.mkdir(exist_ok=True)

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
        return items

    def get_valid_responses(self, row, resp_cols, excl_cols):
        return [float(row[rc]) for ec, rc in zip(excl_cols, resp_cols) if row[ec] != 'True']

    def score_lenient_single(self, v, expected):
        """Same acceptance windows as HumanCIComputer.score_lenient (see
        compute_human_ci.py), but scores every individual rater's raw
        response against the item's expected answer rather than the item's
        pooled mean -- so each item yields a per-rater sequence of 0/1
        "landed in the window" flags. Alpha on those flags measures whether
        raters agree with each other under lenient scoring, not just
        whether the pooled mean happens to land in the window."""
        expected = str(expected).strip().lower()
        if expected == 'certain':
            return 1 if 2.5 <= v <= 3 else 0
        if expected == 'uncertain':
            return 1 if -2.7 <= v <= 2.7 else 0
        if expected == 'certainly not':
            return 1 if -3 <= v <= -2.5 else 0
        raise ValueError(expected)

    def alpha_table(self, items, keys, id_col='n_items'):
        """Ratings are on a continuous -3..+3 scale, and raters per item are
        ragged (one annotator is excluded on exactly half the items), so
        this uses krippendorff_alpha_ragged's NaN-padded reliability_data
        rather than requiring a dense, fixed-size rater matrix."""
        rows = []
        for key in keys:
            sub = [it for it in items if it['_condition'] == key]
            if len(sub) == 0:
                continue
            units = [it['valid_responses'] for it in sub]
            alpha = krippendorff_alpha_ragged(units, metric='interval')
            binned_units = [[self.score_lenient_single(v, it['_expected_answer']) for v in it['valid_responses']]
                             for it in sub]
            alpha_lenient = krippendorff_alpha_ragged(binned_units, metric='nominal')
            ac1_lenient = gwet_ac1_ragged(binned_units, categories=[0, 1])
            n_ratings = sum(len(u) for u in units)
            rows.append((key, len(units), n_ratings, alpha, alpha_lenient, ac1_lenient))
            print(f'{key}: {id_col}={len(units)}, n_ratings={n_ratings}, alpha={alpha:.3f}, '
                  f'alpha_lenient={alpha_lenient:.3f}, ac1_lenient={ac1_lenient:.3f}')
        return rows

    def save_csv(self, path, header, rows):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f'Saved {path.name}')

    def compute(self):
        items = self.load_data()

        print('By condition:')
        by_condition = self.alpha_table(items, self.ALL_CONDITIONS)
        self.save_csv(self._cache_dir / 'krippendorff_by_condition.csv',
                      ['condition', 'n_items', 'n_ratings', 'alpha', 'alpha_lenient', 'ac1_lenient'], by_condition)

        # Pickled for direct consumption by the figure scripts (same pattern
        # as human_ci_cache.pkl / forest_ci_cache.pkl).
        alpha_cache = {cond: alpha for cond, n_items, n_ratings, alpha, alpha_lenient, ac1_lenient in by_condition}
        alpha_lenient_cache = {cond: alpha_lenient for cond, n_items, n_ratings, alpha, alpha_lenient, ac1_lenient in by_condition}
        ac1_lenient_cache = {cond: ac1_lenient for cond, n_items, n_ratings, alpha, alpha_lenient, ac1_lenient in by_condition}
        with open(self._cache_dir / 'krippendorff_alpha_cache.pkl', 'wb') as f:
            pickle.dump(alpha_cache, f)
        with open(self._cache_dir / 'krippendorff_alpha_lenient_cache.pkl', 'wb') as f:
            pickle.dump(alpha_lenient_cache, f)
        with open(self._cache_dir / 'gwet_ac1_lenient_cache.pkl', 'wb') as f:
            pickle.dump(ac1_lenient_cache, f)
        print('Saved krippendorff_alpha_cache.pkl / krippendorff_alpha_lenient_cache.pkl / gwet_ac1_lenient_cache.pkl')


if __name__ == '__main__':
    computer = KrippendorffByConditionComputer()
    computer.compute()
