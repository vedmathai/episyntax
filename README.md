# EpiSyntax

EPISYNTAX is a diagnostic NLP dataset testing whether language models
correctly parse (1) presupposition scope under epistemic markers, and (2)
tracking of sentence repairs (cancellations, reinforcements, weakenings)
of both assertions and presuppositions. It covers 20 scenarios x 13
conditions (8 experiments E1-E8, 5 controls C1-C5) x 2 polarity groups x 2
questions, evaluated through three task framings: Task 1 (certainty
label), Task 2 (scalar -3 to 3), Task 3 (relative comparison between
paired conditions).


## What this repository contains

| # | Item | Location |
|---|------|----------|
| 1 | The dataset (20 scenarios x 13 conditions) | [`data/stimuli/`](data/stimuli/) |
| 2 | The prompts used (canonical + 5 paraphrase variants per task) | [`prompts/`](prompts/) |
| 3 | The prompts given to human annotators | [`human_annotation/instructions/`](human_annotation/instructions/) |
| 4 | The human annotations (anonymized) | [`human_annotation/data/`](human_annotation/data/), [`human_annotation/analysis/`](human_annotation/analysis/) |
| 5 | The code used to run the experiments | [`code/`](code/) |
| 6 | The results from the models | [`results/`](results/) |

### 1. Dataset

- `data/stimuli/Modal_data_corrected.csv` -- the master stimulus sheet:
  all scenario text, all 13 conditions x 2 polarity groups.
- `data/stimuli/full_stimulus_set.csv` -- the same data exploded into
  1,040 individual items (13 conditions x 20 scenarios x 2 polarity
  groups x 2 questions) with `_expected_answer` ground truth attached per
  item. This is the "ready to run experiments on" format used for Task 2.
- `data/stimuli/task1_stimulus_10scenarios.csv`,
  `task3_stimulus_10scenarios.csv` -- the reduced 10-scenario stimulus
  sets used for Tasks 1 and 3, built from `Modal_data_corrected.csv` by
  `code/collection/build_task1_task3_stimuli.py` (regenerating this
  script reproduces both files exactly).

### 2. Prompts

- **Canonical task prompts** and their **5 paraphrased variants each**
  (used for the prompt-robustness bootstrap-CI methodology) live in
  `prompts/`:
  - `prompts/task1_templates.py`, `prompts/task3_templates.py` -- Task 1's
    and Task 3's templates A-E each.
  - `prompts/task2_templates.py` -- templates A-E, all complete.

### 3. Human annotator instructions

- `human_annotation/instructions/task_instructions.md` -- full Prolific
  participant instructions.
- `human_annotation/instructions/inline_instructions.md` -- the compact,
  per-item version of the same instructions.

### 4. Human annotations (anonymized)

- `human_annotation/data/calibration_responses_anonymized.csv` -- the
  calibration-batch raw responses (80 items x 5 raters), anonymized by
  `human_annotation/anonymize_calibration_data.py` from the original
  Prolific export (which is **not** included in this repo).
- `human_annotation/analysis/prolific_calibration_tracking.csv`,
  `prolific_coverage_tracking.csv` -- tracking/scoring sheets with
  `_expected_answer` derived from the answer_map described in
  [Known bug-fix history](#known-bug-fix-history).

**Anonymization method:** the calibration batch had two independent task
groups of 5 raters each, with entirely non-overlapping real annotator
identities. Each group's real Prolific ID hashes were replaced with
sequential `annotator_1`..`annotator_5` labels, assigned independently
per group so the two groups' identities are never conflated under a
shared label space. Timestamps were truncated to date-only. An empty
`Metadata` column (constant `{}` for every row) was dropped as
non-informative.

**The excluded annotator:** one rater, in task group 1, showed a
systematic sign-inverted response pattern (mean absolute deviation 2.61
from the other four raters' mean, vs. 0.72-1.18 for the other four
raters in that group -- computed directly from the raw export). Rather
than silently dropping or silently including this rater's responses, the
anonymized release keeps them and marks them with a per-row
`Annotator{N}_ExcludedFromAnalysis` boolean column, so the exclusion
decision is transparent and reproducible. `compute_human_ci.py` and
`add_human_to_task_comparison.py` both filter on this column (not on a
hardcoded ID) when computing the published human-accuracy figures.

### 5. Code

```
code/
├── collection/
│   ├── multi_model_test_suite.py      Task 1 + Task 3 data collection, 4-model roster
│   └── build_task1_task3_stimuli.py   builds the 10-scenario Task 1/3 stimulus sets
├── analysis/
│   ├── compute_final_ci.py            Task 2 bootstrap CI, condition-level, all 4 models
│   ├── compute_human_ci.py            adds human-annotation bootstrap CI
│   ├── build_task_comparison_v2.py    task-level comparison, pooled across conditions
│   ├── add_human_to_task_comparison.py
│   ├── compute_krippendorff_by_condition.py   human IAA (alpha, lenient alpha, Gwet's AC1), by condition
│   └── iaa_metrics.py                 Krippendorff's alpha, Fleiss' kappa, Gwet's AC1, ICC(3,1)/ICC(3,k)
└── figures/
    ├── render_acl_plots.py            the four main condition-level figures
    └── render_task_comparison.py      the task-level comparison figure
```

All analysis/figure scripts resolve their input/output paths relative to
the repo root (`Path(__file__).resolve().parents[...]`), so they can be
run directly from a checkout with no path editing. See
[Running the pipeline](#running-the-pipeline) below for the full,
ordered command sequence -- from building stimuli, through data
collection, to the final figures.

### 6. Results

```
results/
├── task1/{raw,summary}/   certainty-label task, 10-scenario sample
├── task2/{raw,summary}/   scalar task, full 20-scenario coverage
└── task3/{raw,summary}/   relative-comparison task, 10-scenario sample
```

`raw/` holds one row per (model, item, paraphrase template) API call.
`summary/` holds one row per item, aggregated across the 5 templates
(mean score + lenient-window correctness for Task 2, majority vote +
exact-match correctness for Tasks 1 and 3).

## Running the pipeline

```bash
pip install numpy matplotlib openai krippendorff statsmodels irrCAC pingouin
```

### 1. Build the reduced Task 1/3 stimulus sets

Regenerates `data/stimuli/task1_stimulus_10scenarios.csv` and
`task3_stimulus_10scenarios.csv` from `Modal_data_corrected.csv`. Not
needed unless you want to verify reproducibility or change the sample --
the outputs are already committed.

```bash
python3 code/collection/build_task1_task3_stimuli.py
```

### 2. Collect model responses (Task 1 + Task 3)

`multi_model_test_suite.py` queries all 4 models (see
[Model roster](#model-roster-and-configuration) below) across both
tasks. It's resumable (safe to kill and rerun -- already-completed
(model, item, template) combinations are skipped) and fails loudly on
any non-rate-limit API error rather than silently recording it as data.

```bash
export DO_MODEL_ACCESS_KEY="sk-do-..."   # DigitalOcean Gradient/Inference key

# it uses relative paths, so run it from a scratch directory containing
# copies of its inputs:
mkdir -p /tmp/episyntax_collection && cd /tmp/episyntax_collection
cp /path/to/episyntax/prompts/task1_templates.py .
cp /path/to/episyntax/prompts/task3_templates.py .
cp /path/to/episyntax/data/stimuli/task1_stimulus_10scenarios.csv .
cp /path/to/episyntax/data/stimuli/task3_stimulus_10scenarios.csv .
cp /path/to/episyntax/code/collection/multi_model_test_suite.py .
python3 multi_model_test_suite.py

# once it finishes (or you want a snapshot), move the outputs back in:
cp task1_results_*.csv /path/to/episyntax/results/task1/raw/
cp task3_results_*.csv /path/to/episyntax/results/task3/raw/
```

`results/task2/raw/` is the authoritative source for Task 2's results.

### 3. Compute bootstrap CIs and the task-level comparison

Run in this order -- each step reads a `.pkl` cache written by the
previous one, from a git-ignored `cache/` directory created
automatically on first run:

```bash
python3 code/analysis/compute_final_ci.py             # Task 2 CI, condition-level, all 4 models
python3 code/analysis/build_task_comparison_v2.py      # Task 1+2+3 CI, pooled across conditions
python3 code/analysis/compute_human_ci.py              # human-annotation bootstrap CI
python3 code/analysis/add_human_to_task_comparison.py  # merges human CI into the task-comparison cache
cd code/analysis && python3 compute_krippendorff_by_condition.py && cd ../..  # human IAA, by condition
```

`compute_krippendorff_by_condition.py` imports `iaa_metrics` directly
(not as a package), so it must be run from inside `code/analysis/` rather
than from the repo root.

### 4. Render figures

```bash
python3 code/figures/render_acl_plots.py         # four condition-level ACL figures
python3 code/figures/render_task_comparison.py   # task-level comparison figure
```

Output (PNG + PDF) is written to a git-ignored `output/figures/`
directory.

## Model roster and configuration

| Key | Model ID (DigitalOcean Gradient/Inference) |
|---|---|
| gpt5 | `openai-gpt-5.5` |
| gpt_oss_120b | `openai-gpt-oss-120b` |
| llama70 | `llama3.3-70b-instruct` |
| deepseek | `deepseek-v4-pro` |

All calls use `temperature=0` via an OpenAI-compatible endpoint
(`https://inference.do-ai.run/v1`). Qwen3.5-397B-A17B was evaluated
during model selection but dropped from the final 4-model roster.

## Citation

The paper is not yet public. Once it is, cite it as:

```bibtex
@inproceedings{episyntax,
  title     = {TODO: paper title},
  author    = {TODO: author list},
  booktitle = {TODO: venue},
  year      = {TODO},
  url       = {TODO: arXiv/ACL anthology link}
}
```

## License

Code (`code/`, `prompts/`, `human_annotation/anonymize_calibration_data.py`)
is released under the [MIT License](LICENSE). Data and results (`data/`,
`human_annotation/data/`, `human_annotation/analysis/`,
`human_annotation/instructions/`, `results/`) are released under
[CC BY 4.0](LICENSE-DATA).
