"""
Anonymizes the raw Prolific calibration-batch export before public release.

Input (NOT included in this repository -- contains real Prolific worker ID
hashes, so it never leaves the researcher's machine): the raw wide-format
export with columns Annotator{1..5}_ID / _Response / _Timestamp, one row
per calibration item, for two independent task groups (5 raters each,
non-overlapping between groups).

What this script does:
  1. Replaces each group's 5 real annotator ID hashes with sequential
     anonymous IDs (annotator_1..annotator_5), assigned independently per
     task group so the two groups' raters are never conflated under a
     shared ID space.
  2. Renames the opaque Task_Group_ID UUIDs to readable group labels
     (calibration_group_1, calibration_group_2).
  3. Adds an explicit `excluded_from_analysis` column per annotator slot,
     flagging whichever rater in a group has a mean absolute deviation
     (MAD) from the other 4 raters' responses that exceeds
     EXCLUSION_MAD_THRESHOLD -- this is how the one rater with a
     systematic sign-inverted response pattern was originally identified
     (MAD 2.61 vs. 0.72-1.18 for that group's other four raters). This
     rater's raw responses are KEPT in the release (for transparency/
     reproducibility of the exclusion decision) but flagged rather than
     silently dropped or silently left unexplained.
  4. Truncates timestamps to date-only (drops intra-day precision, which
     is not needed for any analysis in this repo and narrows the
     fine-grained-timing fingerprinting surface).

Re-run against the raw export if it's ever regenerated; do not hand-edit
the anonymized output. Real annotator IDs and group UUIDs never appear in
this script -- the exclusion is detected programmatically, not by a
hardcoded ID, so nothing identifying needs to live in source control.
"""
import csv
from pathlib import Path


class CalibrationAnonymizer:
    # Any rater whose MAD from the other 4 raters' mean response (over
    # items their group shares) exceeds this is flagged as excluded. The
    # known outlier in this dataset has MAD 2.61; every other rater is in
    # the 0.72-1.18 range, so a threshold of 2.0 separates them cleanly.
    EXCLUSION_MAD_THRESHOLD = 2.0
    GROUP_LABEL_PREFIX = 'calibration_group_'
    FRONT_COLS = ['DataPoint_ID', 'Task_Group_ID', 'Task_Type', 'Question', 'Context', 'What Lila said']

    def __init__(self):
        self._repo_root = Path(__file__).resolve().parents[1]
        self._raw_path = self._repo_root / 'human_annotation' / 'data' / 'raw_calibration_responses_NEEDS_ANONYMIZATION.csv'
        self._out_path = self._repo_root / 'human_annotation' / 'data' / 'calibration_responses_anonymized.csv'

    def out_path(self):
        return self._out_path

    def read_csv(self, path):
        with open(path, newline='') as f:
            return list(csv.DictReader(f))

    def label_groups(self, rows):
        group_ids = sorted({row['Task_Group_ID'] for row in rows})
        group_labels = {gid: f'{self.GROUP_LABEL_PREFIX}{i+1}' for i, gid in enumerate(group_ids)}
        for row in rows:
            row['Task_Group_ID'] = group_labels[row['Task_Group_ID']]
        return rows

    def find_excluded_slot(self, group_rows):
        responses = {i: [float(row[f'Annotator{i}_Response']) for row in group_rows] for i in range(1, 6)}
        mad_per_rater = {}
        for i in range(1, 6):
            diffs = []
            for idx in range(len(group_rows)):
                others = [responses[j][idx] for j in range(1, 6) if j != i]
                others_mean = sum(others) / len(others)
                diffs.append(abs(responses[i][idx] - others_mean))
            mad_per_rater[i] = sum(diffs) / len(diffs)
        excluded_slot = max(mad_per_rater, key=mad_per_rater.get)
        if mad_per_rater[excluded_slot] <= self.EXCLUSION_MAD_THRESHOLD:
            return None
        return excluded_slot

    def response_column_is_int(self, rows, i):
        """pandas' to_csv (used to generate the previously-committed
        anonymized file) infers int vs. float per Response column from
        whether ANY value across the whole raw file has a decimal point --
        if so, every value in that column is written with a trailing .0,
        even the ones that were originally integer-valued. Replicated here
        so the plain-csv rewrite produces byte-identical output."""
        return all('.' not in row[f'Annotator{i}_Response'] for row in rows)

    def format_response(self, value, is_int_column):
        if is_int_column:
            return str(int(float(value)))
        return str(float(value))

    def anonymize_group(self, group_rows, int_columns):
        excluded_slot = self.find_excluded_slot(group_rows)

        # Same real rater occupies the same slot across all rows in a group.
        real_ids_in_slot_order = [group_rows[0][f'Annotator{i}_ID'] for i in range(1, 6)]
        anon_map = {real_id: f'annotator_{i+1}' for i, real_id in enumerate(real_ids_in_slot_order)}

        for row in group_rows:
            for i in range(1, 6):
                row[f'Annotator{i}_Response'] = self.format_response(row[f'Annotator{i}_Response'], int_columns[i])
                row[f'Annotator{i}_ExcludedFromAnalysis'] = str(i == excluded_slot)
                row[f'Annotator{i}_ID'] = anon_map[row[f'Annotator{i}_ID']]
                row[f'Annotator{i}_Timestamp'] = row[f'Annotator{i}_Timestamp'][:10]  # date only
        return group_rows

    def per_annotator_cols(self):
        return [
            c for i in range(1, 6)
            for c in (f'Annotator{i}_ID', f'Annotator{i}_Response', f'Annotator{i}_Timestamp', f'Annotator{i}_ExcludedFromAnalysis')
        ]

    def anonymize(self):
        rows = self.read_csv(self._raw_path)
        rows = self.label_groups(rows)

        # Column order (Annotator1_ID..Annotator5_ID) is a fixed, arbitrary
        # per-group assignment -- see anonymize_group.
        int_columns = {i: self.response_column_is_int(rows, i) for i in range(1, 6)}

        groups = {}
        for row in rows:
            groups.setdefault(row['Task_Group_ID'], []).append(row)
        anon_rows = []
        for group_label in sorted(groups):
            anon_rows.extend(self.anonymize_group(groups[group_label], int_columns))
        anon_rows.sort(key=lambda row: row['DataPoint_ID'])

        fieldnames = self.FRONT_COLS + self.per_annotator_cols()
        with open(self._out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            for row in anon_rows:
                writer.writerow({field: row[field] for field in fieldnames})

        n_excluded_rows = sum(
            any(row[f'Annotator{i}_ExcludedFromAnalysis'] == 'True' for row in anon_rows)
            for i in range(1, 6)
        )
        print(f'Wrote {len(anon_rows)} rows to {self._out_path}')
        print(f"Groups: {sorted({row['Task_Group_ID'] for row in anon_rows})}")
        print(f'Excluded rater flagged in {n_excluded_rows} annotator slot(s) (should be exactly 1).')
        return anon_rows


if __name__ == '__main__':
    anonymizer = CalibrationAnonymizer()
    anonymizer.anonymize()
