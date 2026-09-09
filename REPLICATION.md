# Replication fidelity

What was kept identical to the original, what was changed, and why.

This document exists because "I replicated a paper" means nothing without it.
A replication is only as good as its account of where it deviates.

**Status: no experiment has been run.** Every results table below is empty and
marked as such. See `results/NOT_YET_RUN.md`.

---

## 1. What was reproduced exactly

| Component | Source | Fidelity |
|---|---|---|
| **Evaluation metric** | `src/lost_in_the_middle/metrics.py` | **Verbatim port.** `normalize_answer` and `best_subspan_em` are character-for-character identical. See `src/litm2026/scoring.py`. |
| **Prompt templates** | `src/lost_in_the_middle/prompts/*.prompt` | **Vendored unchanged.** Byte-identical files. `prompting.prompt_file_digests()` returns their hashes. |
| **Document format string** | `prompting.py` | **Identical**: `Document [{i+1}](Title: {title}) {text}`, 1-indexed. |
| **Multi-document QA data** | `qa_data/` | **Used as released.** 2,655 examples per file; 10/20/30 document configurations; gold positions 0/4/9/14/19 for 20 documents. Downloaded, not regenerated. |
| **Key-value retrieval data** | `kv_retrieval_data/` | **Used as released.** 75/140/300 keys. |
| **Decoding** | Paper §3 | Greedy: `temperature=0.0`, `top_p=1.0`, `max_new_tokens=100`. |
| **Baselines** | Paper §2.3 | Both retained: closed-book (no documents) and oracle (gold document only). |

The verbatim port is deliberate, and the reasoning is worth stating: an
"improved" metric produces numbers that cannot be compared to the paper's. The
comparison *is* the experiment.

---

## 2. What was changed, and why

Every deviation, with its justification and its risk.

### 2.1 Chat-format adaptation — the most significant deviation

**What changed.** The paper's models (GPT-3.5-Turbo aside) were prompted in
completion style: a single string ending in `Answer:`, with the model expected
to continue it. The models tested here are chat models with a messages API.
The completion-style prompt has to be placed inside a message.

**How it is handled.** `prompting.to_chat_messages()` implements four
adaptations, selectable per run via `chat_adaptation` in the config:

| Mode | What it does |
|---|---|
| `user_verbatim` (default) | The paper's exact prompt string as a single user message, `Answer:` cue included. Closest to the original. |
| `user_no_answer_cue` | Same, with the trailing `Answer:` removed — chat models often do not need it. |
| `system_instruction` | Instruction moved to a system message. |
| `user_verbatim_terse` | Verbatim plus a brevity instruction. |

**Why `user_verbatim` is the default.** It changes the least. The `Answer:` cue
is preserved even though it is arguably redundant for a chat model, because
removing it is a change and keeping it is not.

**The risk, stated plainly.** If the U-shape is sensitive to prompt formatting,
this adaptation could inflate or suppress it. That is not a hypothetical
concern — it is the most likely alternative explanation for any divergence from
the paper.

**The mitigation, which you must actually run:** A/B the adaptations on a
subset and report the spread. A result that survives all four modes is far
stronger than one measured under a single template.

```bash
for mode in user_verbatim user_no_answer_cue system_instruction; do
  python -m litm2026.runner --config experiments/configs/main_qa.yaml \
    --override chat_adaptation=$mode --override run_id=ab_$mode --override n=50
done
```

> **Fill in after running.** If the A/B was not run, say so here rather than
> leaving the impression that it was.

| Adaptation | U-shape severity | 95% CI |
|---|---|---|
| `user_verbatim` | NOT YET RUN | — |
| `user_no_answer_cue` | NOT YET RUN | — |
| `system_instruction` | NOT YET RUN | — |

### 2.2 Sample size: 150 per cell, not 2,655

**What changed.** The paper used all 2,655 NQ-open queries per cell. This uses
150 by default, for cost — the full matrix at full n across three models is far
beyond a free-tier budget.

**Why it is defensible, and what it costs.**
- The **same 150 questions** appear in every cell (seeded via
  `data.select_question_subset`), so positions are compared **pairwise**. The
  paired design recovers much of the power a smaller n would otherwise cost.
- `stats.mde_paired_binary(150)` gives the **minimum detectable effect**. Report
  it beside the result. If the observed effect is smaller than the MDE, the
  honest conclusion is "underpowered", not "no effect".
- Every number carries a bootstrap CI, so the imprecision is visible rather than
  implied.

**What it does not fix.** 150 is 150. Small effects and between-model
differences that the paper could resolve may be invisible here.

### 2.3 Different models

The paper tested GPT-3.5-Turbo, Claude-1.3, GPT-4, MPT-30B-Instruct and
LongChat-13B-16K. None are current. This tests models available on free tiers in
2026 — which is the point of the exercise, but it means **no model is common to
both studies**. Comparisons are between *eras*, not between matched systems.

### 2.4 Closed-weight models behind opaque endpoints

The models are served through APIs. A provider can change the model behind a
stable name without notice. This result is a **snapshot with a date on it**,
not a permanent property. Record the run date and any model version string the
provider returns.

### 2.5 Original comparison numbers are read from figures

The paper reports positional results as plots, not tables. Values in
`data/original_results.csv` must be **read off the published figures by hand**
and are approximate. The file ships empty, deliberately — fill it yourself from
the paper rather than trusting recalled numbers.

---

## 3. Where my numbers match the original

> **NOT YET RUN.** Fill in after `main_qa.yaml` completes.

| Configuration | Paper (2023) | This replication (2026) | Δ |
|---|---|---|---|
| 20 docs, gold at 0 | — | — | — |
| 20 docs, gold at 9 (middle) | — | — | — |
| 20 docs, gold at 19 | — | — | — |
| Closed-book | — | — | — |
| Oracle | — | — | — |
| **U-shape severity** | — | — | — |

**The specific claim to check:** the paper found that at some configurations,
middle-position accuracy fell **below the closed-book baseline** — the model did
worse with the relevant document buried than with no documents at all. Does that
still happen?

| Model | Middle-position accuracy | Closed-book | Below floor? |
|---|---|---|---|
| — | NOT YET RUN | — | — |

---

## 4. Where they diverge, and why they might

> **NOT YET RUN.**

When you fill this in, work through the candidate explanations rather than
picking one. For each, say whether your data can rule it out:

| Explanation | Can this study distinguish it? |
|---|---|
| **Long-context training** — models are now explicitly trained on long inputs | Partly. A flattening consistent across labs points this way. |
| **Chat adaptation artefact** — the deviation in §2.1 | **Only if you ran the A/B.** Otherwise this is not ruled out. |
| **Data contamination** — NQ is public and old; answers may be memorised | Partly. Compare against the synthetic KV task, which cannot be memorised. |
| **Subsample variance** — n=150 | Yes, via the CIs and the MDE. |
| **Instruction tuning** — better instruction-following on "use the documents" | No. Would need matched base/instruct pairs. |
| **Tokenizer and context-length differences** | No. |

Do not assert a mechanism this experiment cannot test. "My experiment measures
the behaviour, not the mechanism" is the correct and defensible position.

---

## 5. Threats to validity

Ordered by how much they should worry you.

### 5.1 Data contamination — the most serious

Natural Questions is public and predates every model tested. Some answers may be
**memorised from pre-training rather than retrieved from context**. A model
answering from memory would show a flat positional curve for reasons that have
nothing to do with long-context ability — which means contamination could
manufacture the very "the effect has gone away" result that looks like a finding.

**Partial mitigation:** the synthetic key-value task uses random UUIDs that
cannot have been memorised. If QA flattens but KV does not, contamination is the
leading explanation.

**Raise this yourself before anyone asks.**

### 5.2 Single prompt template

Effect size may be prompt-sensitive. Without the §2.1 A/B, prompt formatting is
not ruled out as the cause of any divergence.

### 5.3 Statistical power

n=150 per cell. Small effects will not be resolved. Report the MDE.

### 5.4 Model opacity and drift

Closed models, versioned by the provider, changeable without notice.

### 5.5 Discrete positions

Five positions sampled, not a continuum. A narrow dip between sampled points
would be missed.

### 5.6 Single dataset

NQ-open only. The paper's finding may generalise differently on other retrieval
corpora.

### 5.7 Free-tier serving

Free-tier endpoints may quantize, route, or throttle differently from paid
tiers. The model that answered may not be the model you think you tested.

---

## 6. What would falsify my conclusion

The section that matters most, and the one almost no portfolio replication has.

**If the conclusion is "the U-shape persists":**
- A run at n≥1000 showing the effect within noise → I was reading variance.
- The effect vanishing under a different prompt template → prompt artefact, not
  a model property.
- The effect appearing only in QA and never in synthetic KV → likely a property
  of the dataset or its distractors, not of positional attention.

**If the conclusion is "the U-shape has flattened":**
- A decontaminated dataset restoring the U-shape → I measured memorisation.
- The effect reappearing at longer contexts than tested → it moved rather than
  disappeared, and my range was too narrow.
- Same models at paid tier showing the U-shape → a serving artefact.

**In either case:**
- Failure to reproduce my numbers from my own committed raw responses would mean
  a bug in the analysis, not a finding. Every figure must be recomputable from
  `results/raw/` with no API key.

---

## Citation

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

Thanks to the authors for releasing their data, prompts and evaluation code
under a permissive licence. This replication exists because they did.
