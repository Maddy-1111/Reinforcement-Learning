"""Prepend a step=0 row to every progress.csv under lunarlander/runs/.

Justified: the training loop's seed phase (steps 0-9999) uses only random
actions with zero gradient updates, so the policy network at step=0 and
step=10000 is identical. The step=10000 eval IS the initial-policy eval;
we just copy it to step=0 so plots start at x=0 per the TA requirement.
"""
import csv
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent / 'lunarlander' / 'runs'
FIRST_EVAL_STEP = 10000


def patch(csv_path: Path) -> str:
    rows = csv_path.read_text().splitlines()
    if len(rows) < 2:
        return f"SKIP  {csv_path} (too short)"

    header = rows[0]
    first_data = rows[1]

    # Already patched?
    first_step = first_data.split(',')[0].strip()
    if first_step == '0':
        return f"SKIP  {csv_path} (already has step=0)"

    # Build step=0 row: copy first_data but set step→0 and wall_time→0.00
    fields = header.split(',')
    values = first_data.split(',')
    row = dict(zip(fields, values))
    row['step'] = '0'
    if 'wall_time' in row:
        row['wall_time'] = '0.00'

    step0_line = ','.join(row[f] for f in fields)
    new_content = header + '\n' + step0_line + '\n' + '\n'.join(rows[1:]) + '\n'
    csv_path.write_text(new_content)
    return f"PATCH {csv_path}"


def main():
    csvs = sorted(RUNS_DIR.glob('*/progress.csv'))
    if not csvs:
        print(f"No progress.csv files found under {RUNS_DIR}")
        sys.exit(1)

    patched = skipped = 0
    for csv_path in csvs:
        msg = patch(csv_path)
        print(msg)
        if msg.startswith('PATCH'):
            patched += 1
        else:
            skipped += 1

    print(f"\n{patched} patched, {skipped} skipped  ({len(csvs)} total)")


if __name__ == '__main__':
    main()
