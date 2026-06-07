# Research Log — Prompting vs. Model Scaling

A record of what was investigated, how it was carried out, and the decisions made along the way.
Author: Samir Samal (American High School).

## 1. Research question

When a fixed budget can be spent on either a **bigger model** or a **better prompt**, which wins —
and does the answer depend on the task? The study frames this quantitatively as a
**prompt–parameter exchange rate**: how many parameters of scaling a prompting strategy substitutes
for.

## 2. Experimental design

A full factorial sweep over four dimensions:

| Dimension | Values |
|---|---|
| Model family | Qwen2.5-Instruct, Llama-3 |
| Size | Qwen: 0.5B / 1.5B / 3B / 7B · Llama: 1B / 3B / 8B |
| Task | SST-2 (sentiment), MMLU (knowledge, 57 subjects), GSM8K (multi-step math) |
| Prompting strategy | zero-shot, few-shot, chain-of-thought (CoT), structured (JSON) |

- Same family at different sizes, with **4-bit quantization held constant**, so parameter count is
  the only within-family variable.
- **n = 300 examples per condition**, **two random seeds (42, 43)**, pooled to **n = 600**.
- Approximately **50,000 model generations** in total.

## 3. Methodology (how it was run)

- **Local inference via MLX** on an Apple M4 laptop (16 GB unified memory), one model resident at a
  time. (Initially attempted on Kaggle with `transformers` + bitsandbytes, but 4-bit bitsandbytes is
  CUDA-only and the full grid would not finish in a session — the work was moved to MLX locally.)
- **Chat templates** applied to every prompt, since instruction-tuned models expect their chat
  format; omitting it depresses and distorts results.
- **Greedy decoding (temperature 0)** — deterministic and reproducible; measures the model's best
  single answer rather than sampling luck.
- **Per-condition output limits** (8 tokens for a multiple-choice letter, up to 512 for GSM8K CoT so
  reasoning chains are not truncated).
- **Task-specific answer extraction** (regex + JSON parsing); scored by numeric match for GSM8K and
  exact label/letter match otherwise. Extraction failures count as incorrect.
- Results **saved incrementally** after each model, so partial progress survives interruptions.

## 4. Statistics

- **95% Wilson score confidence intervals** — chosen because several accuracies sit near 0%, where
  the standard normal-approximation interval misbehaves.
- **Paired McNemar tests**, continuity-corrected and **Holm-adjusted** for multiple comparisons, to
  test whether each strategy genuinely differs from zero-shot on the same examples.

## 5. Robustness and verification

1. **Metric-artifact check (Schaeffer 2023 critique).** On GSM8K, a continuous signal
   (whether the correct value appears anywhere in the output) and the answer's relative error were
   also recorded. These track the exact-match scores rather than diverging, so the reasoning results
   are not an artifact of a discontinuous metric.
2. **MMLU-CoT extraction audit.** The "CoT hurts knowledge" result was re-run storing full
   (untruncated) outputs; the extraction-failure rate was only 2–4%, and the effect held — confirming
   it is real rather than a parsing failure.
3. **Cross-seed stability.** Across seeds 42 and 43, condition accuracies differed by ~2.3pp on
   average (max ~7.7pp), within the confidence intervals, and every finding replicated.
4. **Cross-family generality.** Running the second model family caught an over-generalization: two
   findings replicated across families; one turned out to be model-specific.

## 6. Literature grounding

A multi-source literature scan identified the ~13 closest prior works (chain-of-thought, scaling
laws, emergent abilities and the "mirage" critique, format restrictions, prompt-format sensitivity).
Every cited arXiv identifier was opened and verified; one author misattribution was caught and
corrected.

## 7. Findings

1. **The exchange rate is large on reasoning and near-zero on knowledge** — the opposite of the
   intuition that prompting helps weak models most. On GSM8K, CoT at a small size beats zero-shot at
   a much larger size; on MMLU, no prompting strategy meaningfully beats scaling.
2. **Chain-of-thought is double-edged, and the cost is model-specific.** CoT unlocks arithmetic but
   *degrades* multiple-choice knowledge QA in Qwen (−7 to −17pp, worse with scale), while remaining
   neutral-to-helpful in Llama. Reported as family-dependent, not universal.
3. **Structured output and few-shot are task- and family-dependent**, not universal penalties or
   benefits.

## 8. How the work actually unfolded (decision log)

Real research is iterative; the design was refined as problems surfaced:

- Moved from a Kaggle/CUDA setup to **local MLX** when the full grid could not finish.
- Fixed an early bug: prompts were not wrapped in the **chat template**.
- Fixed an **MMLU few-shot** prompt that initially contained no exemplars.
- Adjusted **per-condition token limits** so CoT reasoning was not cut off.
- Added the **continuous GSM8K metric** to pre-empt the metric-artifact critique.
- Ran the **Qwen ladder** (overnight; survived a system-sleep stall and a battery shutdown thanks to
  incremental saving).
- **Audited the MMLU-CoT** result with full-output storage.
- Ran the **Llama ladder** for cross-family generality.
- Added **seed 43** for both families to estimate variance.
- Built the figures, manuscript, appendices, and this repository.

## Reproducing

See [README.md](README.md). In short: install requirements, run `run_experiments_local.py` for each
family/seed, then `--aggregate` to rebuild every table, and `make_figures.py` for the figures.

## AI assistance disclosure

Experimental code, statistical analysis, figure generation, and an initial manuscript draft were
produced with the assistance of an AI coding assistant; this is disclosed in the paper's
Acknowledgements. See the paper for the full statement.
