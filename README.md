# Lost in the Middle, replicated on 2026 models

**Does a 2023 finding about long-context retrieval still hold on models that advertise a million tokens?**

A faithful replication of [Liu et al., *Lost in the Middle*](https://arxiv.org/abs/2307.03172)
(TACL 2024) — the authors' data, the authors' metric, the authors' prompts —
extended to current models and to context lengths the original could not reach.

<!-- HEADLINE CHART GOES HERE once `main_qa.yaml` has run.
     results/figures/usi_vs_context_length.png
     This is the first thing anyone sees. Do not bury it below the fold. -->

> ### ⚠️ No experiment has been run yet
> This repository is **code-complete and result-empty**. Every table below is
> marked `NOT YET RUN`, and there are **no placeholder numbers anywhere** — a
> plausible invented figure is worse than a blank, because a blank cannot be
> mistaken for a finding. See [`results/NOT_YET_RUN.md`](results/NOT_YET_RUN.md)
> to produce results.

---

## TL;DR

- Replicates the positional-bias ("lost in the middle") effect using the
  original released data, the original `best_subspan_em` metric ported
  verbatim, and the original prompt templates vendored unchanged.
- Adds what the 2023 study could not test: **does the effect persist, flatten,
  or move** on 2026 long-context models, and what happens past 100k tokens.
- Reports bootstrap confidence intervals, paired significance tests, and a
  minimum detectable effect — because a positional accuracy curve without error
  bars is decoration, not evidence.

**Headline result:** `NOT YET RUN`

---

## The original finding

Give a model *k* documents and a question, where exactly one document contains
the answer and the rest are topically similar distractors. Sweep the position of
that gold document and measure accuracy.

Accuracy is **U-shaped**: high at the start, high at the end, lowest in the
middle. In the paper's sharpest result, middle-position accuracy fell **below
the closed-book baseline** — the model performed worse with the relevant
document buried in context than with no documents at all.

That was measured on models with 4k–16k context windows.

## The question this repository asks

Models now advertise 200k to 1M+ tokens and are explicitly trained on long
inputs. **Is the U-shape gone?**

The popular answer is "yes, needle-in-a-haystack tests are saturated." That
answer conflates two different tasks. Needle-in-a-haystack hides one *unrelated*
fact in filler — nothing else in the context looks like an answer. This task uses
19 real passages *about the same topic*, so the model must discriminate, not
merely locate. **A model can score 100% on needle-in-a-haystack and still show a
strong U-shape here.**

---

## Setup

| | |
|---|---|
| **Data** | The authors' released `qa_data/` (2,655 NQ-open examples per file) and `kv_retrieval_data/` |
| **Configurations** | 10, 20 and 30 documents; gold positions 0/4/9/14/19 for 20 docs |
| **Baselines** | Closed-book (no documents) **and** oracle (gold only) — both, so the curve has a scale |
| **Metric** | `best_subspan_em`, ported verbatim from the original |
| **Decoding** | Greedy, `temperature=0`, `max_new_tokens=100` |
| **n per cell** | **150** (the original used 2,655 — reduced for cost, see below) |
| **Statistics** | Bootstrap CIs (10,000 resamples), McNemar's exact test, paired bootstrap, Holm–Bonferroni |

**On n=150, stated up front rather than buried:** this is a real limitation.
Two things make it workable — the *same* 150 questions run in every cell, so
comparisons are paired rather than independent; and `stats.mde_paired_binary`
reports the minimum detectable effect, which is published alongside the result.
Underpowering an experiment is acceptable. Hiding it is not.

---

## Results

> `NOT YET RUN` — run `experiments/configs/main_qa.yaml`.

### Accuracy by gold-document position (20 documents)

| Model | Pos 0 | Pos 4 | Pos 9 | Pos 14 | Pos 19 | Closed-book | Oracle | U-shape severity |
|---|---|---|---|---|---|---|---|---|
| Llama 3.3 70B | — | — | — | — | — | — | — | — |
| Llama 3.1 8B | — | — | — | — | — | — | — | — |
| Mistral Small | — | — | — | — | — | — | — | — |

All cells to carry 95% bootstrap CIs.

### 2023 vs 2026

| | Paper (2023) | This replication (2026) |
|---|---|---|
| U-shape severity, 20 docs | — | — |
| Middle position below closed-book? | — | — |

### Does the effect return at longer contexts?

| Context length | U-shape severity |
|---|---|
| ~75 KV pairs | — |
| ~140 KV pairs | — |
| ~300 KV pairs | — |

**U-shape severity** is defined as `(mean(first, last) − min(middle)) / mean(first, last)`.
Zero means flat; higher means a deeper dip; negative means an inverted curve.

---

## What this means if you are building RAG

The practical payoff, independent of which way the result goes:

1. **Ranking and placement are two decisions, not one.** Your retriever returns
   an order. That is not necessarily the order to put in the prompt.
2. **More retrieved context is not monotonically better.** There is a point
   where adding documents costs accuracy. Find it for your setup rather than
   filling the window because it is there.
3. **Reranking pays twice** — better documents selected, and better positioning
   of them.
4. **If you must include many passages, put the highest-confidence one last**,
   nearest the question.

---

## Reproduce it

```bash
git clone https://github.com/Arnavdsp/lost-in-the-middle-2026.git
cd lost-in-the-middle-2026

pip install -e ".[dev]"
python -m pytest                      # 84 tests, no API key needed

python scripts/download_data.py       # ~257 MB from the authors' repo
cp .env.example .env                  # add one key: Groq, Mistral, anything OpenAI-compatible

# 1. Pilot first. n=5. Costs cents. Catches every pipeline bug.
python -m litm2026.runner --config experiments/configs/pilot.yaml

# 2. Read raw responses BY EYE before trusting any aggregate.
#    A pipeline returning well-formed garbage looks fine in a summary table.
head -3 results/raw/pilot/*.jsonl | python -m json.tool

# 3. The main run. Resumable — re-run the same command if it dies.
python -m litm2026.runner --config experiments/configs/main_qa.yaml

# 4. The long-context extension.
python -m litm2026.runner --config experiments/configs/kv_longcontext.yaml

# 5. Statistics and figures.
python -m litm2026.stats  --run-dir results/raw/main_qa
python -m litm2026.plotting --summary results/summary.csv
```

**Set a spend limit in your provider console before the first real run**, not
just the `budget:` block in the config. Belt and braces.

Every raw API response is written to `results/raw/`. Once a run exists it is
committed, so **every figure is recomputable without an API key** — a number in
this README whose underlying response is not in the repository would be an
assertion, not a result.

---

## Limitations

The full account is in **[REPLICATION.md](REPLICATION.md)**. The three that
matter most:

1. **Data contamination.** Natural Questions is public and older than every
   model tested. Some answers may be recalled rather than retrieved, which could
   manufacture a flat curve that looks like a finding. The synthetic key-value
   task exists partly to bound this.
2. **A chat-format adaptation was necessary.** The paper's models were
   completion models. `to_chat_messages()` makes the adaptation explicit and
   A/B-able, but a single template is still a single template.
3. **The models are closed and served behind mutable endpoints.** This is a
   dated snapshot, not a permanent property of these systems.

---

## Repository layout

```
src/litm2026/
  scoring.py     # best_subspan_em -- VERBATIM port, do not "improve"
  prompting.py   # original templates vendored + documented chat adaptation
  data.py        # loaders; the seeded subset that makes cells paired
  providers.py   # Anthropic + any OpenAI-compatible endpoint
  runner.py      # async, resumable, cost-capped
  stats.py       # bootstrap CIs, McNemar, Holm, MDE, U-shape severity
  plotting.py    # refuses to draw a curve without confidence intervals
experiments/configs/   # every run is a config file; no hardcoded parameters
results/raw/           # every API response -- the evidence behind every number
```

---

## Credit and licence

Built on [nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle)
(MIT, © 2023 Nelson Liu). The data, prompts and metric are the authors' work,
used as released; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). This
replication is possible only because they published their materials.

```bibtex
@article{liu-etal-2024-lost,
  title   = {Lost in the Middle: How Language Models Use Long Contexts},
  author  = {Liu, Nelson F. and Lin, Kevin and Hewitt, John and
             Paranjape, Ashwin and Bevilacqua, Michele and
             Petroni, Fabio and Liang, Percy},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2024},
  url     = {https://arxiv.org/abs/2307.03172}
}
```

This repository's own code: MIT.
