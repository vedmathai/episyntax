"""
Multi-model test suite for gpt-5.5, gpt-oss-120b, Llama-3.3-70B, and
DeepSeek V4 Pro via DigitalOcean's Gradient/Inference platform
(OpenAI-compatible endpoint).

RESUMABLE: results are written to disk after EVERY single call. If the
connection drops, the process is killed, or you stop it, rerunning picks up
exactly where it left off -- already-completed (model, item, template)
combinations are skipped, no re-querying.

Setup:
    pip install openai
    export DO_MODEL_ACCESS_KEY="sk-do-..."

This script uses relative paths and expects to be run from a working
directory containing (copy or symlink these in before running):
    task1_templates.py                 (../../prompts/task1_templates.py)
    task3_templates.py                 (../../prompts/task3_templates.py)
    task1_stimulus_10scenarios.csv     (../../data/stimuli/task1_stimulus_10scenarios.csv)
    task3_stimulus_10scenarios.csv     (../../data/stimuli/task3_stimulus_10scenarios.csv)
Output (task1_results_{model}.csv, task3_results_{model}.csv) is written
to that same working directory; move the finished files into
results/task1/raw/ and results/task3/raw/ once collection completes.
"""
import csv
import os
import time
from collections import deque
from openai import OpenAI, RateLimitError
from task1_templates import (
    TASK1_TEMPLATE_A, TASK1_TEMPLATE_B, TASK1_TEMPLATE_C, TASK1_TEMPLATE_D, TASK1_TEMPLATE_E,
)
from task3_templates import (
    TASK3_TEMPLATE_A, TASK3_TEMPLATE_B, TASK3_TEMPLATE_C, TASK3_TEMPLATE_D, TASK3_TEMPLATE_E,
)


class MultiModelTestSuite:
    MODEL_IDS = {
        'gpt5': 'openai-gpt-5.5',
        'gpt_oss_120b': 'openai-gpt-oss-120b',
        'llama70': 'llama3.3-70b-instruct',
        'deepseek': 'deepseek-v4-pro',
    }
    TASK1_TEMPLATES = {
        'A': TASK1_TEMPLATE_A, 'B': TASK1_TEMPLATE_B, 'C': TASK1_TEMPLATE_C,
        'D': TASK1_TEMPLATE_D, 'E': TASK1_TEMPLATE_E,
    }
    TASK3_TEMPLATES = {
        'A': TASK3_TEMPLATE_A, 'B': TASK3_TEMPLATE_B, 'C': TASK3_TEMPLATE_C,
        'D': TASK3_TEMPLATE_D, 'E': TASK3_TEMPLATE_E,
    }
    VALID_LABELS = {'certain', 'uncertain', 'certainly not'}
    VALID_CHOICES = {'situation 1', 'situation 2'}
    WINDOW_SIZE = 20

    MAX_RATE_LIMIT_RETRIES = 8
    INITIAL_BACKOFF_SECONDS = 30
    BACKOFF_MULTIPLIER = 1.5
    MAX_BACKOFF_SECONDS = 300  # cap so it doesn't grow unbounded on a long outage

    def __init__(self):
        self._client = OpenAI(base_url='https://inference.do-ai.run/v1')

    def client(self):
        return self._client

    def read_csv(self, path):
        with open(path, newline='') as f:
            return list(csv.DictReader(f))

    def list_available_models(self):
        models = self._client.models.list()
        for m in models.data:
            print(m.id)

    def call_model(self, model_id, prompt, max_tokens, task_label):
        backoff = self.INITIAL_BACKOFF_SECONDS
        for attempt in range(1, self.MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=model_id, max_tokens=max_tokens, temperature=0,
                    messages=[{'role': 'user', 'content': prompt}],
                )
            except RateLimitError as e:
                if attempt == self.MAX_RATE_LIMIT_RETRIES:
                    print(f'  Rate limit still hit after {self.MAX_RATE_LIMIT_RETRIES} attempts -- giving up.')
                    raise
                print(f'  Rate limited (attempt {attempt}/{self.MAX_RATE_LIMIT_RETRIES}) on {task_label}: {e}. '
                      f'Waiting {backoff:.0f}s before retrying...')
                time.sleep(backoff)
                backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)

    def query_task1(self, model_id, template_str, context, statement, question, max_tokens=1200):
        q_clean = question.replace('How certain are you that ', '').rstrip('?')
        prompt = template_str.format(context=context, statement=statement, question=q_clean)
        response = self.call_model(model_id, prompt, max_tokens, task_label='Task 1')

        text = response.choices[0].message.content
        if text is None:
            finish_reason = response.choices[0].finish_reason
            raise ValueError(f'content is None for {model_id} (finish_reason={finish_reason!r}) on Task 1.')
        text = text.strip().lower().strip('.')
        if text not in self.VALID_LABELS:
            raise ValueError(f'Non-label response from {model_id} on Task 1: {text!r} '
                              f'(expected one of {self.VALID_LABELS})')
        return text

    def query_task3(self, model_id, template_str, context, question, statement_1, statement_2, max_tokens=1200):
        q_clean = question.replace('How certain are you that ', '').rstrip('?')
        prompt = template_str.format(context=context, question=q_clean,
                                      statement_1=statement_1, statement_2=statement_2)
        response = self.call_model(model_id, prompt, max_tokens, task_label='Task 3')

        text = response.choices[0].message.content
        if text is None:
            finish_reason = response.choices[0].finish_reason
            raise ValueError(f'content is None for {model_id} (finish_reason={finish_reason!r}) on Task 3.')
        text = text.strip().lower().strip('.')
        if text not in self.VALID_CHOICES:
            raise ValueError(f'Non-choice response from {model_id} on Task 3: {text!r} '
                              f'(expected one of {self.VALID_CHOICES})')
        return text

    def load_completed_keys(self, output_path, key_cols):
        if not os.path.exists(output_path):
            return set()
        rows = self.read_csv(output_path)
        return {tuple(row[c] for c in key_cols) for row in rows}

    def append_row(self, output_path, row_dict, fieldnames):
        file_exists = os.path.exists(output_path)
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
            f.flush()
            os.fsync(f.fileno())

    def format_duration(self, seconds):
        hours = seconds / 3600
        if hours < 1:
            return f'{seconds/60:.0f} min'
        return f'{hours:.1f} hr'

    def report_progress(self, model_key, task_label, i, total, new_calls_made, remaining_total, call_timestamps):
        if new_calls_made % 20 != 0:
            return
        remaining = remaining_total - new_calls_made
        if len(call_timestamps) >= 2:
            window_elapsed = call_timestamps[-1] - call_timestamps[0]
            window_calls = len(call_timestamps) - 1
            rate = window_calls / window_elapsed if window_elapsed > 0 else None
        else:
            rate = None
        if rate and rate > 0 and remaining > 0:
            eta_seconds = remaining / rate
            rate_per_min = rate * 60
            print(f'  [{model_key}] {task_label}: {i}/{total}  '
                  f'(recent rate: {rate_per_min:.1f} calls/min over last '
                  f'{len(call_timestamps)} calls, est. remaining: {self.format_duration(eta_seconds)})')
        else:
            print(f'  [{model_key}] {task_label}: {i}/{total}  (gathering rate estimate...)')

    def run_task1_test(self, model_key, model_id, stimulus_rows):
        """Resumable, fail-loud. Prints a moving-window ETA based on the last
        WINDOW_SIZE completed calls, not the overall average since start --
        this avoids the estimate being thrown off by early rate-limit waits
        or a slow first few calls."""
        output_path = f'task1_results_{model_key}.csv'
        fieldnames = ['model', 'scenario_idx', 'condition', 'polarity_group', 'q_used',
                      'expected_answer', 'template', 'raw_answer']
        key_cols = ['model', 'scenario_idx', 'condition', 'polarity_group', 'q_used', 'template']

        completed = self.load_completed_keys(output_path, key_cols)
        n_already_done = len(completed)
        print(f'  [{model_key}] Task 1: {n_already_done} already completed, resuming...')

        total = len(stimulus_rows) * len(self.TASK1_TEMPLATES)
        remaining_total = total - n_already_done
        i = 0
        new_calls_made = 0
        call_timestamps = deque(maxlen=self.WINDOW_SIZE)

        for row in stimulus_rows:
            for tkey, tstr in self.TASK1_TEMPLATES.items():
                key = (model_key, row['_scenario_idx'], row['_condition'], row['_polarity_group'],
                       row['_q_used'], tkey)
                i += 1
                if key in completed:
                    continue

                raw = self.query_task1(model_id, tstr, row['Context'], row['What Lila said'], row['Question'])
                call_timestamps.append(time.time())
                new_calls_made += 1

                self.append_row(output_path, {
                    'model': model_key, 'scenario_idx': row['_scenario_idx'], 'condition': row['_condition'],
                    'polarity_group': row['_polarity_group'], 'q_used': row['_q_used'],
                    'expected_answer': row['_expected_answer'], 'template': tkey, 'raw_answer': raw,
                }, fieldnames)
                time.sleep(0.1)
                self.report_progress(model_key, 'Task 1', i, total, new_calls_made, remaining_total, call_timestamps)

        print(f'  [{model_key}] Task 1 complete.')

    def run_task3_test(self, model_key, model_id, stimulus_rows):
        """Same moving-window ETA approach as run_task1_test."""
        output_path = f'task3_results_{model_key}.csv'
        fieldnames = ['model', 'scenario_idx', 'pair', 'condition_a', 'condition_b',
                      'expected_choice', 'template', 'raw_answer']
        key_cols = ['model', 'scenario_idx', 'pair', 'template']

        completed = self.load_completed_keys(output_path, key_cols)
        n_already_done = len(completed)
        print(f'  [{model_key}] Task 3: {n_already_done} already completed, resuming...')

        total = len(stimulus_rows) * len(self.TASK3_TEMPLATES)
        remaining_total = total - n_already_done
        i = 0
        new_calls_made = 0
        call_timestamps = deque(maxlen=self.WINDOW_SIZE)

        for row in stimulus_rows:
            for tkey, tstr in self.TASK3_TEMPLATES.items():
                key = (model_key, row['_scenario_idx'], row['_pair'], tkey)
                i += 1
                if key in completed:
                    continue

                raw = self.query_task3(model_id, tstr, row['Context'], row['Question'],
                                        row['statement_1'], row['statement_2'])
                call_timestamps.append(time.time())
                new_calls_made += 1

                self.append_row(output_path, {
                    'model': model_key, 'scenario_idx': row['_scenario_idx'], 'pair': row['_pair'],
                    'condition_a': row['_condition_a'], 'condition_b': row['_condition_b'],
                    'expected_choice': row['_expected_choice'], 'template': tkey, 'raw_answer': raw,
                }, fieldnames)
                time.sleep(0.1)
                self.report_progress(model_key, 'Task 3', i, total, new_calls_made, remaining_total, call_timestamps)

        print(f'  [{model_key}] Task 3 complete.')

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
        sorted groupby would, so the output ordering is reproducible.
        scenario_idx is coerced to int so it sorts numerically, not
        lexicographically."""
        groups = {}
        for row in rows:
            key = tuple(int(row[c]) if c == 'scenario_idx' else row[c] for c in item_cols)
            groups.setdefault(key, []).append(row)
        return {key: groups[key] for key in sorted(groups)}

    def analyze(self, results_path, summary_path, item_cols, expected_col, group_label):
        """Accuracy: majority vote across the 5 templates per item (exact
        label/choice match required, no lenient windows since output is
        already a discrete label)."""
        rows = self.read_csv(results_path)
        groups = self.item_groups(rows, item_cols)

        summary_fieldnames = item_cols + ['raw_answer', 'majority_answer', 'correct']
        summary_rows = []
        for key, group_rows in groups.items():
            answers = [row['raw_answer'] for row in group_rows]
            majority_answer = self.mode(answers)
            expected = key[item_cols.index(expected_col)]
            summary_rows.append(dict(zip(item_cols, key), **{
                'raw_answer': answers, 'majority_answer': majority_answer,
                'correct': int(majority_answer == expected),
            }))

        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(summary_rows)

        group_col = item_cols[1]
        by_group = {}
        for row in summary_rows:
            by_group.setdefault(row[group_col], []).append(row['correct'])
        print(f'\n=== {group_label} ===')
        for g in sorted(by_group):
            corrects = by_group[g]
            print(f'{g}: mean={sum(corrects)/len(corrects):.3f}, count={len(corrects)}')
        return summary_rows

    def analyze_task1(self, model_key):
        item_cols = ['scenario_idx', 'condition', 'polarity_group', 'q_used', 'expected_answer']
        return self.analyze(
            f'task1_results_{model_key}.csv', f'task1_summary_{model_key}.csv', item_cols, 'expected_answer',
            f'{model_key}: Task 1 accuracy (majority vote across 5 templates), by condition',
        )

    def analyze_task3(self, model_key):
        item_cols = ['scenario_idx', 'pair', 'condition_a', 'condition_b', 'expected_choice']
        return self.analyze(
            f'task3_results_{model_key}.csv', f'task3_summary_{model_key}.csv', item_cols, 'expected_choice',
            f'{model_key}: Task 3 accuracy (majority vote across 5 templates), by pair',
        )

    def run(self):
        task1_stimulus = self.read_csv('task1_stimulus_10scenarios.csv')
        task3_stimulus = self.read_csv('task3_stimulus_10scenarios.csv')

        for model_key, model_id in self.MODEL_IDS.items():
            print(f'\n=== Running {model_key} ({model_id}) ===')
            self.run_task1_test(model_key, model_id, task1_stimulus)
            self.analyze_task1(model_key)
            self.run_task3_test(model_key, model_id, task3_stimulus)
            self.analyze_task3(model_key)

        print('\nAll models complete for Tasks 1 and 3. Per-model files: '
              'task1_results_<model>.csv / task1_summary_<model>.csv, '
              'task3_results_<model>.csv / task3_summary_<model>.csv.')


if __name__ == '__main__':
    # Model IDs are confirmed against your account. Still worth an auth
    # sanity check before the full run if it's been a while since you last
    # verified the key works:
    # MultiModelTestSuite().list_available_models()
    # raise SystemExit

    suite = MultiModelTestSuite()
    suite.run()
