import csv
from pathlib import Path


class StimuliBuilder:
    CONDITIONS = {1: 'E2', 2: 'C2', 3: 'E1', 4: 'C1', 5: 'C3', 6: 'E3', 7: 'E4',
                  8: 'E5', 9: 'C4', 10: 'C5', 11: 'E6', 12: 'E7', 13: 'E8'}
    SAMPLE_SCENARIOS = set(range(1, 11))

    # Pairs are (experiment_condition, baseline_condition). Per Appendix B's
    # logic: if the experiment's expected answer is 'certain', pair with a
    # baseline whose expected answer is 'uncertain' (and vice versa),
    # avoiding ambiguous ties.
    TASK3_PAIRS = [
        ('E1', 'C4'), ('E2', 'C4'), ('E3', 'C4'),
        ('E5', 'C5'), ('E6', 'C4'), ('E7', 'C4'), ('E8', 'C5'),
    ]
    RANK = {'certain': 2, 'uncertain': 1, 'certainly not': 0}

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[2]
        self._stimuli_dir = self._repo_root / 'data' / 'stimuli'
        self._scenarios = None
        self._answer_map = None
        self._cond_to_num = {v: k for k, v in self.CONDITIONS.items()}

    def stimuli_dir(self):
        return self._stimuli_dir

    def load_data(self):
        with open(self._stimuli_dir / 'Modal_data_corrected.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        # rows[0] is a descriptive-labels row, not scenario data.
        scenarios = rows[1:]
        self._scenarios = [row for row in scenarios if int(float(row['Idx'])) in self.SAMPLE_SCENARIOS]
        self._answer_map = self.build_answer_map()
        return self._scenarios

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

    def write_csv(self, path, fieldnames, rows):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def build_task1(self):
        fieldnames = ['Context', 'What Lila said', 'Question', '_scenario_idx', '_condition',
                      '_polarity_group', '_q_used', '_expected_answer']
        rows = []
        for scenario in self._scenarios:
            idx = int(float(scenario['Idx']))
            context = scenario['S1']
            for cond_num, cond_label in self.CONDITIONS.items():
                for polarity_group in ['C1', 'C2']:
                    for q in ['Q1', 'Q2']:
                        sentence = scenario[f'{polarity_group}.{cond_num}']
                        question_text = scenario[q]
                        expected = self._answer_map[f'{polarity_group}.{cond_num}_{q}']
                        rows.append({
                            'Context': context, 'What Lila said': sentence, 'Question': question_text,
                            '_scenario_idx': idx, '_condition': cond_label,
                            '_polarity_group': polarity_group, '_q_used': q, '_expected_answer': expected,
                        })
        self.write_csv(self._stimuli_dir / 'task1_stimulus_10scenarios.csv', fieldnames, rows)
        print(f'Task 1 stimulus: {len(rows)} items ({len(self.SAMPLE_SCENARIOS)} scenarios x 13 conditions x 2 groups x 2 Qs)')
        return rows

    def build_task3(self):
        fieldnames = ['Context', 'Question', 'statement_1', 'statement_2', '_scenario_idx', '_pair',
                      '_condition_a', '_condition_b', '_expected_choice']
        rows = []
        for scenario in self._scenarios:
            idx = int(float(scenario['Idx']))
            context = scenario['S1']
            for cond_a_label, cond_b_label in self.TASK3_PAIRS:
                cond_a_num = self._cond_to_num[cond_a_label]
                cond_b_num = self._cond_to_num[cond_b_label]
                # C1 polarity group / Q1 for both, for a clean single
                # comparison per scenario.
                statement_1 = scenario[f'C1.{cond_a_num}']
                statement_2 = scenario[f'C1.{cond_b_num}']
                question_text = scenario['Q1']
                expected_a = self._answer_map[f'C1.{cond_a_num}_Q1']
                expected_b = self._answer_map[f'C1.{cond_b_num}_Q1']
                if self.RANK[expected_a] > self.RANK[expected_b]:
                    expected_choice = 'situation 1'
                elif self.RANK[expected_b] > self.RANK[expected_a]:
                    expected_choice = 'situation 2'
                else:
                    expected_choice = 'tie -- excluded'
                rows.append({
                    'Context': context, 'Question': question_text,
                    'statement_1': statement_1, 'statement_2': statement_2,
                    '_scenario_idx': idx, '_pair': f'{cond_a_label}_vs_{cond_b_label}',
                    '_condition_a': cond_a_label, '_condition_b': cond_b_label,
                    '_expected_choice': expected_choice,
                })
        n_ties = sum(1 for row in rows if row['_expected_choice'] == 'tie -- excluded')
        print(f'Task 3 stimulus: {len(rows)} items ({len(self.SAMPLE_SCENARIOS)} scenarios x {len(self.TASK3_PAIRS)} pairs)')
        print(f'  -- {n_ties} ties found (ambiguous, should be excluded from scoring)')
        self.write_csv(self._stimuli_dir / 'task3_stimulus_10scenarios.csv', fieldnames, rows)
        return rows

    def build(self):
        self.load_data()
        self.build_task1()
        self.build_task3()


if __name__ == '__main__':
    builder = StimuliBuilder()
    builder.build()
