# No experiment in this repository has been run

This file exists so that the absence of results is **explicit** rather than
something a reader has to infer from empty directories.

- `results/raw/` is empty. No API call has been made.
- `results/figures/` is empty. No figure has been generated from real data.
- Every results table in `README.md` and `REPLICATION.md` is marked
  `NOT YET RUN`.
- **There are no placeholder numbers anywhere in this repository.** Not in the
  README, not in the docstrings, not as "illustrative examples". A plausible
  invented number is worse than a blank, because a blank cannot be mistaken
  for a finding.

## To produce results

```bash
pip install -e ".[dev]"
python scripts/download_data.py
cp .env.example .env          # then add at least one API key

# 1. Pilot first. n=5. Costs cents. Catches every pipeline bug.
python -m litm2026.runner --config experiments/configs/pilot.yaml

# 2. READ some raw responses by eye before trusting any aggregate.
#    A pipeline that returns well-formed garbage looks fine in a summary table.
head -3 results/raw/pilot/*.jsonl | python -m json.tool

# 3. The main run.
python -m litm2026.runner --config experiments/configs/main_qa.yaml

# 4. The long-context extension.
python -m litm2026.runner --config experiments/configs/kv_longcontext.yaml

# 5. Statistics and figures.
python -m litm2026.stats --run-dir results/raw/main_qa
python -m litm2026.plotting --summary results/summary.csv
```

## When results exist

Delete this file, and commit the raw responses:

```bash
git add -f results/raw/main_qa
```

The raw responses are the evidence. A README number whose underlying response
is not in the repository is an assertion, not a result — anyone should be able
to recompute every figure from `results/raw/` without an API key.
