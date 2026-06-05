# Prompting Substitutes for Scale on Reasoning, but Not on Knowledge

Code, prompts, and results for a controlled study of the **prompt–parameter exchange rate** —
how much model scaling a prompting strategy can substitute for, and how that depends on the task.

**Author:** Samir Samal (American High School)

## Summary

We sweep two instruction-tuned model families — **Qwen2.5** (0.5B / 1.5B / 3B / 7B) and
**Llama-3** (1B / 3B / 8B), 4-bit quantization held constant — across four prompting strategies
(**zero-shot, few-shot, chain-of-thought, structured/JSON**) on three tasks: **SST-2** (sentiment),
**MMLU** (knowledge), and **GSM8K** (multi-step arithmetic). Everything runs locally on a single
16 GB Apple-silicon laptop via [MLX](https://github.com/ml-explore/mlx). Two random seeds (42, 43),
n = 300 examples per condition (pooled n = 600), with Wilson confidence intervals and
Holm-adjusted McNemar tests.

## Key findings

- **The exchange rate is largest on reasoning, smallest on knowledge** — the opposite of the naive
  intuition. On GSM8K, chain-of-thought at a small size beats zero-shot at a much larger size
  (Qwen-1.5B+CoT 55.7% > Qwen-7B zero-shot 14.3%; Llama-3B+CoT 74.7% > Llama-8B zero-shot 22.7%).
  On MMLU, no strategy improves on zero-shot by more than a few points — scaling is the only lever.
- **CoT is double-edged, and its cost is model-specific.** In Qwen, CoT *degrades* multiple-choice
  knowledge QA by 6–17 pp (worse with scale); in Llama, CoT is neutral-to-helpful. Reported as
  family-dependent, not universal.
- Effects are robust to the scoring-artifact critique (continuous numeric-presence metric;
  re-extraction from full outputs) and replicate across both seeds and both families.

## Repository layout

```
run_experiments_local.py   # main harness: models, prompts, extraction, scoring, stats, --aggregate
make_figures.py            # Figures 1–3 from results/aggregate_accuracy.csv
make_appendix.py           # Appendix C tables (Wilson CIs + Holm-adjusted McNemar), splices into paper
reorder_jei.py             # produce a JEI-ordered (Methods-last) variant of the paper
results/                   # per-example CSVs (seeds 42/43, both families) + aggregated tables
figures/                   # generated figures
paper.pdf                  # the write-up
```

All prompt templates live in `build_prompt()` in `run_experiments_local.py` and are also listed in
Appendix A of `paper.pdf`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt
```

Requires an Apple-silicon Mac for the local (MLX) models.

## Reproduce

```bash
# Qwen ladder (seed 42); add --seed 43 for the second seed:
python run_experiments_local.py --models qwen-0.5b qwen-1.5b qwen-3b qwen-7b
# Llama ladder:
python run_experiments_local.py --models llama-1b llama-3b llama-8b --tag llama
# Build the aggregated tables (merges every results/results_seed_*.csv):
python run_experiments_local.py --aggregate
# Figures:
python make_figures.py
```

Optional flags: `--benchmarks`, `--strategies`, `--n_samples`, `--seed`, `--tag`. An optional
`GEMINI_API_KEY` environment variable enables a Gemini reference model; it is never stored in code.

## License

MIT — see [LICENSE](LICENSE).
