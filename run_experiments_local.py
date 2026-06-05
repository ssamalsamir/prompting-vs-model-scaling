"""
run_experiments_local.py — Prompt engineering vs. model scaling (Apple Silicon)
================================================================================
Thesis: "How many parameters is a good prompt worth?"

Runs locally on an Apple Silicon Mac via MLX (Metal-accelerated). Two same-family
ladders give a GENERALITY check across model families:
  - Qwen2.5-Instruct: 0.5B / 1.5B / 3B / 7B  (all 4-bit)
  - Llama-3:          1B / 3B / 8B            (all 4-bit)
Gemini-2.5-Flash is a separate-family "instruction-following ceiling".

Why this differs from a CUDA/Kaggle setup:
  - Backend is mlx-lm, NOT transformers+bitsandbytes (4-bit bnb is CUDA-only).
  - Defaults to n=300, ONE seed. Decoding is greedy, so a seed only ever varied
    the eval subset (sampling noise) — already captured by Wilson CIs + paired
    McNemar. One fixed 300-item set is cleaner and gives more paired power.
  - Gemini "thinking" is DISABLED: otherwise hidden reasoning tokens bill at the
    output rate AND can exhaust max_output_tokens, returning empty answers.

SETUP (local, one time)
-----------------------
  python3 -m venv ~/.venvs/promptscaling
  source ~/.venvs/promptscaling/bin/activate
  pip install -U mlx-lm datasets pandas scipy tqdm google-genai
  export GEMINI_API_KEY="your-key-here"   # only if including Gemini

RUN (run each family on its own night so the Mac/runtime stays light; --tag
keeps the output files separate, and --aggregate merges everything):
  # Night 1 — Qwen ladder + Gemini:
  caffeinate -i python run_experiments_local.py
  # Night 2 — Llama ladder (second family for the generality claim):
  caffeinate -i python run_experiments_local.py --models llama-1b llama-3b llama-8b --tag llama
  # Build the thesis tables (merges ALL results_seed_*.csv):
  python run_experiments_local.py --aggregate

NOTE: Llama models may be gated on the Hub. The mlx-community 4-bit repos used
here are normally ungated; if a download 401s, run `huggingface-cli login` once.
"""

import os, json, time, argparse, random, re, gc, shutil, math, glob
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# ── Environment ───────────────────────────────────────────────────────────────

def setup_env():
    """Return a Gemini API key from Kaggle secrets (if present) or the env."""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        try:
            return secrets.get_secret("GEMINI_API_KEY")
        except Exception:
            pass
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        print("✅ GEMINI_API_KEY found in environment.")
    else:
        print("ℹ️  No GEMINI_API_KEY (local models will still run).")
    return key

# ── Model Configuration ───────────────────────────────────────────────────────
# Within each family, only parameter count varies AND 4-bit quant is held
# constant -> clean scaling comparison.

GPU_COST_PER_HOUR = 0.0   # local Mac compute is "free"; the thesis leans on the
                          # hardware-independent metric (tokens_per_correct).

MODEL_CONFIGS = {
    # ── Qwen2.5 ladder ──
    "qwen-0.5b": {"hf_name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "type": "local",
                  "params_b": 0.5, "family": "Qwen2.5-Instruct"},
    "qwen-1.5b": {"hf_name": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "type": "local",
                  "params_b": 1.5, "family": "Qwen2.5-Instruct"},
    "qwen-3b":   {"hf_name": "mlx-community/Qwen2.5-3B-Instruct-4bit",   "type": "local",
                  "params_b": 3.0, "family": "Qwen2.5-Instruct"},
    "qwen-7b":   {"hf_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",   "type": "local",
                  "params_b": 7.0, "family": "Qwen2.5-Instruct"},
    # ── Llama-3 ladder (second family: generality check) ──
    "llama-1b":  {"hf_name": "mlx-community/Llama-3.2-1B-Instruct-4bit",       "type": "local",
                  "params_b": 1.0, "family": "Llama-3"},
    "llama-3b":  {"hf_name": "mlx-community/Llama-3.2-3B-Instruct-4bit",       "type": "local",
                  "params_b": 3.0, "family": "Llama-3"},
    "llama-8b":  {"hf_name": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",  "type": "local",
                  "params_b": 8.0, "family": "Llama-3"},
    # ── Separate-family ceiling ──
    "gemini-2.5-flash": {"name": "gemini-2.5-flash", "type": "api", "params_b": None,
                         "family": "Gemini", "input_cost_per_1m": 0.30,
                         "output_cost_per_1m": 2.50},
}

BENCHMARKS = ["sst2", "gsm8k", "mmlu"]
STRATEGIES = ["zero_shot", "few_shot", "cot", "structured"]
N_SAMPLES  = 300   # ±~6pp Wilson CI; plenty for the scaling-vs-prompting claims

SST2_FEW_SHOT_EXAMPLES = [
    ("The film is a masterpiece of visual storytelling.", "positive"),
    ("I fell asleep twice. Complete waste of time.",      "negative"),
    ("A quietly moving portrait of grief and resilience.","positive"),
]

GSM8K_COT_EXAMPLES = [
    {"question": "Janet has 3 cats. Each cat eats 2 cans of food per day. How many cans does she need for a week?",
     "cot": "Each cat eats 2 cans/day. 3 cats eat 3 x 2 = 6 cans/day. In 7 days: 6 x 7 = 42 cans.",
     "answer": "42"},
    {"question": "A store sells apples for $0.50 each. Tom buys 8 apples and pays with a $10 bill. How much change?",
     "cot": "8 x $0.50 = $4.00. Change = $10.00 - $4.00 = $6.00.",
     "answer": "6"},
]

# MMLU few-shot exemplars (answers spread A/B/C/D to avoid position bias).
MMLU_FEW_SHOT_EXAMPLES = [
    {"question": "Which organelle is the primary site of ATP production in eukaryotic cells?",
     "choices": ["Mitochondria", "Nucleus", "Ribosome", "Golgi apparatus"], "answer": "A"},
    {"question": "Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
     "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], "answer": "B"},
    {"question": "What is the derivative of x^2 with respect to x?",
     "choices": ["x", "x^2", "2x", "2"], "answer": "C"},
    {"question": "Who is the author of the play 'Hamlet'?",
     "choices": ["Charles Dickens", "Mark Twain", "Jane Austen", "William Shakespeare"], "answer": "D"},
]

# ── Per-condition generation length ───────────────────────────────────────────

def max_new_tokens_for(benchmark: str, strategy: str) -> int:
    if strategy == "cot":
        return 512 if benchmark == "gsm8k" else 256
    if benchmark == "gsm8k":
        return 48
    if benchmark == "sst2":
        return 16
    return 8   # mmlu: a single letter

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(strategy: str, benchmark: str, example: dict) -> str:
    if benchmark == "sst2":
        t = example["sentence"]
        if strategy == "zero_shot":
            return (f"Classify the sentiment of the following sentence as exactly "
                    f"'positive' or 'negative'.\n\nSentence: {t}\nSentiment:")
        if strategy == "few_shot":
            shots = "\n".join(f"Sentence: {s}\nSentiment: {l}"
                              for s, l in SST2_FEW_SHOT_EXAMPLES)
            return (f"Classify the sentiment as 'positive' or 'negative'.\n\n"
                    f"{shots}\n\nSentence: {t}\nSentiment:")
        if strategy == "cot":
            return (f"Identify the key sentiment signals in the sentence, reason "
                    f"through them, then classify as 'positive' or 'negative'. "
                    f"End your response with 'Final answer: positive' or "
                    f"'Final answer: negative'.\n\nSentence: {t}\nReasoning:")
        if strategy == "structured":
            return (f'Classify sentiment. Respond ONLY with valid JSON: '
                    f'{{"sentiment": "positive"}} or {{"sentiment": "negative"}}.'
                    f'\n\nSentence: {t}\nJSON:')

    if benchmark == "gsm8k":
        q = example["question"]
        if strategy == "zero_shot":
            return (f"Solve the following math problem. "
                    f"Give only the final number as your answer.\n\nProblem: {q}\nAnswer:")
        if strategy == "few_shot":
            shots = "\n\n".join(f"Problem: {e['question']}\nAnswer: {e['answer']}"
                                for e in GSM8K_COT_EXAMPLES)
            return f"Solve the following math problems.\n\n{shots}\n\nProblem: {q}\nAnswer:"
        if strategy == "cot":
            shots = "\n\n".join(
                f"Problem: {e['question']}\nReasoning: {e['cot']}\nAnswer: {e['answer']}"
                for e in GSM8K_COT_EXAMPLES)
            return (f"Solve the following math problems step by step. "
                    f"End with 'Answer: <number>'.\n\n{shots}\n\nProblem: {q}\nReasoning:")
        if strategy == "structured":
            return (f'Solve this math problem. Respond ONLY with valid JSON: '
                    f'{{"answer": "<number>"}}.\n\nProblem: {q}\nJSON:')

    if benchmark == "mmlu":
        q, c = example["question"], example["choices"]
        opts = "\n".join(f"{l}. {v}" for l, v in zip("ABCD", c))
        if strategy == "zero_shot":
            return (f"Answer the following multiple choice question. "
                    f"Respond with only the letter A, B, C, or D.\n\n"
                    f"Question: {q}\n{opts}\nAnswer:")
        if strategy == "few_shot":
            shots = "\n\n".join(
                "Question: {q}\n{o}\nAnswer: {a}".format(
                    q=e["question"],
                    o="\n".join(f"{l}. {v}" for l, v in zip("ABCD", e["choices"])),
                    a=e["answer"])
                for e in MMLU_FEW_SHOT_EXAMPLES)
            return (f"Answer the following multiple choice questions. "
                    f"Respond with only the letter A, B, C, or D.\n\n"
                    f"{shots}\n\nQuestion: {q}\n{opts}\nAnswer:")
        if strategy == "cot":
            return (f"Answer the following multiple choice question. Think through "
                    f"each option carefully, then end your response with "
                    f"'Final answer: A', 'Final answer: B', 'Final answer: C', "
                    f"or 'Final answer: D'.\n\nQuestion: {q}\n{opts}\nReasoning:")
        if strategy == "structured":
            return (f'Answer the following multiple choice question. Respond ONLY '
                    f'with valid JSON: {{"answer": "A"}} (or B, C, D).\n\n'
                    f'Question: {q}\n{opts}\nJSON:')

    return "Prompt error"

# ── Answer Extraction ─────────────────────────────────────────────────────────

def extract_answer(output: str, benchmark: str, strategy: str) -> Optional[str]:
    if not output:
        return None
    text = output.strip()
    text_lower = text.lower()

    if benchmark == "sst2":
        if strategy == "structured":
            try:
                clean = re.sub(r"```json|```", "", text).strip()
                return json.loads(clean).get("sentiment", "").lower()
            except Exception:
                pass
        if strategy == "cot":
            m = re.search(r"final answer[:\s]+(\w+)", text_lower)
            if m and m.group(1) in ("positive", "negative"):
                return m.group(1)
        tail = text_lower[-60:]
        for word in ("positive", "negative"):
            if word in tail:
                return word
        for word in ("positive", "negative"):
            if word in text_lower:
                return word
        return None

    if benchmark == "gsm8k":
        if strategy == "structured":
            try:
                clean = re.sub(r"```json|```", "", text).strip()
                return str(json.loads(clean).get("answer", "")).strip()
            except Exception:
                pass
        m = re.search(r"[Aa]nswer[:\s]+\$?(-?[\d,]+\.?\d*)", text)
        if m:
            return m.group(1).replace(",", "")
        m = re.search(r"=\s*\$?(-?[\d,]+\.?\d*)\s*$", text)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(r"-?[\d,]+\.?\d*", text)
        if nums:
            filtered = [n for n in nums if n.replace(",", "") not in
                        ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]]
            if filtered:
                return filtered[-1].replace(",", "")
        return None

    if benchmark == "mmlu":
        if strategy == "structured":
            try:
                clean = re.sub(r"```json|```", "", text).strip()
                return json.loads(clean).get("answer", "").upper()
            except Exception:
                pass
        if strategy == "cot":
            m = re.search(r"final answer[:\s]+([a-dA-D])", text)
            if m:
                return m.group(1).upper()
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            m = re.search(r"\b([A-D])\b", lines[-1].upper())
            if m:
                return m.group(1)
        matches = re.findall(r"\b([A-D])\b", text.upper())
        if matches:
            return matches[-1]
        return None

    return None


def check_correct(pred: Optional[str], gold: str, benchmark: str) -> bool:
    if pred is None:
        return False
    if benchmark == "gsm8k":
        try:
            return abs(float(str(pred).replace(",", "")) -
                       float(re.sub(r"[^\d.\-]", "", str(gold)))) < 0.01
        except Exception:
            return False
    return str(pred).strip().lower() == str(gold).strip().lower()


def gsm8k_continuous(output: str, pred: Optional[str], gold: str) -> tuple:
    """Robust/continuous signals for GSM8K, to counter the metric-artifact
    critique (Schaeffer et al. 2023) that exact-match exaggerates 'collapse'.
    Returns (numeric_present, rel_error):
      - numeric_present: 1.0 if the correct value appears ANYWHERE in the output
        (a softer 'did it reach the answer' signal than committed exact-match).
      - rel_error: relative error of the committed prediction (continuous).
    Both NaN for non-numeric tasks."""
    try:
        g = float(re.sub(r"[^\d.\-]", "", str(gold)))
    except Exception:
        return (float("nan"), float("nan"))
    vals = []
    for nstr in re.findall(r"-?[\d,]+\.?\d*", output or ""):
        try:
            vals.append(float(nstr.replace(",", "")))
        except Exception:
            pass
    present = 1.0 if any(abs(v - g) < 0.01 for v in vals) else 0.0
    rel = float("nan")
    if pred is not None:
        try:
            rel = abs(float(str(pred).replace(",", "")) - g) / max(abs(g), 1.0)
        except Exception:
            rel = float("nan")
    return (present, rel)

# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_ci(correct: int, n: int, confidence: float = 0.95) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0, (centre - margin)*100), min(100, (centre + margin)*100))


def mcnemar_test(correct_a: list, correct_b: list) -> tuple:
    b = sum(1 for a, x in zip(correct_a, correct_b) if a == 1 and x == 0)
    c = sum(1 for a, x in zip(correct_a, correct_b) if a == 0 and x == 1)
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1)**2 / (b + c)   # continuity-corrected
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return (chi2, p)


def holm_correction(pvals: list) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0]*m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def sig_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

# ── Model Wrappers ────────────────────────────────────────────────────────────

class MLXModel:
    def __init__(self, key: str, cfg: dict):
        from mlx_lm import load
        print(f"\n  Loading {key} ({cfg['hf_name']})...")
        self.model, self.tokenizer = load(cfg["hf_name"])
        # Greedy decoding (temp=0). API has shifted across mlx-lm versions, so
        # detect the sampler interface once and fall back to the temp= kwarg.
        self._sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler
            self._sampler = make_sampler(temp=0.0)
        except Exception:
            self._sampler = None
        print("    Loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 128) -> tuple:
        from mlx_lm import generate
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        in_len = len(self.tokenizer.encode(formatted))
        kwargs = dict(max_tokens=max_new_tokens, verbose=False)
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        else:
            kwargs["temp"] = 0.0   # older mlx-lm
        try:
            text = generate(self.model, self.tokenizer, prompt=formatted, **kwargs)
        except TypeError:
            text = generate(self.model, self.tokenizer, formatted,
                            max_tokens=max_new_tokens, temp=0.0, verbose=False)
        out_len = len(self.tokenizer.encode(text))
        return text, in_len, out_len

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
        print("    Unloaded.")


class GeminiModel:
    def __init__(self, cfg: dict, api_key: str):
        from google import genai
        self.client     = genai.Client(api_key=api_key)
        self.model_name = cfg["name"]
        self.in_cost    = cfg["input_cost_per_1m"]
        self.out_cost   = cfg["output_cost_per_1m"]
        self.total_in_tok  = 0
        self.total_out_tok = 0
        self.total_cost    = 0.0
        print(f"  Initialized Gemini: {self.model_name} (thinking disabled)")

    def _config(self, max_new_tokens: int):
        from google.genai import types
        kwargs = dict(max_output_tokens=max(max_new_tokens, 64), temperature=0.0)
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        return types.GenerateContentConfig(**kwargs)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> tuple:
        time.sleep(0.5)
        cfg = self._config(max_new_tokens)
        for attempt in range(3):
            try:
                res = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=cfg)
                text = res.text or ""
                if hasattr(res, "usage_metadata") and res.usage_metadata:
                    in_t  = getattr(res.usage_metadata, "prompt_token_count", 0) or 0
                    out_t = getattr(res.usage_metadata, "candidates_token_count", 0) or 0
                    self.total_in_tok  += in_t
                    self.total_out_tok += out_t
                    self.total_cost    += (in_t/1e6)*self.in_cost + (out_t/1e6)*self.out_cost
                    return text, in_t, out_t
                return text, 0, 0
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 60 * (attempt + 1)
                    print(f"\n  Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  Gemini error: {e}")
                    return "", 0, 0
        return "", 0, 0

    def unload(self):
        print(f"\n  Gemini running cost: ${self.total_cost:.4f} "
              f"({self.total_in_tok:,} in / {self.total_out_tok:,} out tokens)")

# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_benchmark(benchmark: str, n: int, seed: int) -> list:
    from datasets import load_dataset
    if benchmark == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        exs = [{"sentence": r["sentence"],
                "answer": "positive" if r["label"] == 1 else "negative"} for r in ds]
    elif benchmark == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        exs = [{"question": r["question"],
                "answer": re.sub(r"[^\d.\-]", "", r["answer"].split("####")[-1].strip())}
               for r in ds]
    elif benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        by_subj = {}
        for r in ds:
            by_subj.setdefault(r["subject"], []).append({
                "question": r["question"], "choices": r["choices"],
                "answer": "ABCD"[r["answer"]]})
        rng = random.Random(seed)
        exs, per_subj = [], max(1, n // len(by_subj) + 1)
        for subj_exs in by_subj.values():
            rng.shuffle(subj_exs)
            exs.extend(subj_exs[:per_subj])
        rng.shuffle(exs)
        return exs[:n]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    random.seed(seed)
    random.shuffle(exs)
    return exs[:n]

# ── Main Experiment ───────────────────────────────────────────────────────────

def run_experiment(models_list, benchmarks_list, n_samples, seed, gemini_key,
                   gpu_cost, tag="", strategies_list=None):
    strategies_list = strategies_list or STRATEGIES
    random.seed(seed)
    np.random.seed(seed)
    Path("results").mkdir(exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    records = []

    print(f"\n{'='*65}\nLocal experiment (MLX) — seed={seed}, n={n_samples} per condition")
    print(f"Models:     {models_list}\nBenchmarks: {benchmarks_list}\n{'='*65}\n")

    for benchmark in benchmarks_list:
        print(f"\n── {benchmark.upper()} ──")
        examples = load_benchmark(benchmark, n_samples, seed)
        n_eval = len(examples)
        print(f"   Loaded {n_eval} examples.")

        model_strategy_correct = {}   # for McNemar (all models share these items)

        for m_key in models_list:
            if m_key not in MODEL_CONFIGS:
                print(f"   Unknown model '{m_key}' — skipping.")
                continue
            cfg = MODEL_CONFIGS[m_key]

            if cfg["type"] == "api":
                if not gemini_key:
                    print(f"   Skipping {m_key} — no GEMINI_API_KEY.")
                    continue
                model = GeminiModel(cfg, gemini_key)
            else:
                model = MLXModel(m_key, cfg)

            for strategy in strategies_list:
                max_tok = max_new_tokens_for(benchmark, strategy)
                correct_vec = []
                for ex in tqdm(examples, desc=f"  {m_key}/{strategy}", leave=False):
                    prompt = build_prompt(strategy, benchmark, ex)
                    t0 = time.time()
                    output, in_tok, out_tok = model.generate(prompt, max_tok)
                    elapsed = time.time() - t0

                    pred = extract_answer(output, benchmark, strategy)
                    is_corr = int(check_correct(pred, ex["answer"], benchmark))
                    correct_vec.append(is_corr)

                    if benchmark == "gsm8k":
                        num_present, rel_err = gsm8k_continuous(output, pred, ex["answer"])
                    else:
                        num_present, rel_err = float("nan"), float("nan")

                    records.append({
                        "model": m_key, "family": cfg.get("family", ""),
                        "params_b": cfg.get("params_b"), "benchmark": benchmark,
                        "strategy": strategy, "seed": seed, "correct": is_corr,
                        "numeric_present": num_present, "rel_error": rel_err,
                        "elapsed_sec": round(elapsed, 3),
                        "in_tokens": in_tok, "out_tokens": out_tok,
                        "output": output[:2000],  # keep enough to AUDIT extraction (CoT tails)
                    })

                model_strategy_correct[(m_key, strategy)] = correct_vec
                n_corr = sum(correct_vec)
                acc = n_corr / n_eval * 100
                lo, hi = wilson_ci(n_corr, n_eval)
                print(f"   {m_key:18s} / {strategy:12s}: {acc:5.1f}%  "
                      f"95% CI [{lo:.1f}, {hi:.1f}]")

            model.unload()
            pd.DataFrame(records).to_csv(f"results/results_seed_{seed}{sfx}.csv", index=False)

        # McNemar pairwise tests for local models, Holm-corrected.
        print(f"\n   McNemar pairwise tests ({benchmark}, Holm-corrected):")
        pairs, labels = [], []
        for m_key in models_list:
            if m_key not in MODEL_CONFIGS or MODEL_CONFIGS[m_key]["type"] == "api":
                continue
            for s1, s2 in [("zero_shot", "cot"), ("zero_shot", "few_shot"),
                           ("zero_shot", "structured")]:
                k1, k2 = (m_key, s1), (m_key, s2)
                if k1 in model_strategy_correct and k2 in model_strategy_correct:
                    _, pval = mcnemar_test(model_strategy_correct[k1],
                                           model_strategy_correct[k2])
                    pairs.append(pval)
                    labels.append(f"{m_key:18s} {s1} vs {s2}")
        if pairs:
            for lbl, p_adj in zip(labels, holm_correction(pairs)):
                print(f"   {lbl}: p_adj={p_adj:.4f} {sig_stars(p_adj)}")

    df = pd.DataFrame(records)
    df.to_csv(f"results/results_seed_{seed}{sfx}.csv", index=False)

    print("\n\n── ACCURACY SUMMARY (mean %) ──")
    summary = (df.groupby(["model", "benchmark", "strategy"])["correct"]
                 .mean().mul(100).round(1))
    print(summary.unstack(["benchmark", "strategy"]).to_string())

    g = df[df["benchmark"] == "gsm8k"]
    if len(g):
        print("\n── GSM8K CONTINUOUS METRICS (exact-match vs robust signals) ──")
        cont = g.groupby(["model", "strategy"]).agg(
            exact_match=("correct", "mean"),
            numeric_present=("numeric_present", "mean"),
            median_rel_error=("rel_error", "median"))
        cont[["exact_match", "numeric_present"]] = (cont[["exact_match", "numeric_present"]]*100).round(1)
        cont["median_rel_error"] = cont["median_rel_error"].round(3)
        print(cont.to_string())

    write_cost_table(df, seed, gpu_cost, tag)
    shutil.make_archive(f"experiment_local_seed_{seed}{sfx}", "zip", "results")
    print(f"✅ Zipped → experiment_local_seed_{seed}{sfx}.zip")


def write_cost_table(df, seed, gpu_cost, tag=""):
    sfx = f"_{tag}" if tag else ""
    rows = []
    for (m_key, bench, strat), grp in df.groupby(["model", "benchmark", "strategy"]):
        cfg = MODEL_CONFIGS.get(m_key, {})
        n_corr, n_total = grp["correct"].sum(), len(grp)
        acc = n_corr / n_total * 100
        lo, hi = wilson_ci(int(n_corr), n_total)
        avg_sec = grp["elapsed_sec"].mean()
        avg_tok = (grp["in_tokens"] + grp["out_tokens"]).mean()
        tok_per_correct = (avg_tok * n_total / n_corr) if n_corr else float("nan")

        if cfg.get("type") == "api":
            c1k = ((grp["in_tokens"].mean()/1e6)*cfg.get("input_cost_per_1m", 0) +
                   (grp["out_tokens"].mean()/1e6)*cfg.get("output_cost_per_1m", 0)) * 1000
        else:
            c1k = (avg_sec/3600) * gpu_cost * 1000
        eff = (acc/c1k) if c1k > 0 else float("nan")

        rows.append({
            "model": m_key, "family": cfg.get("family", ""),
            "params_b": cfg.get("params_b"), "benchmark": bench, "strategy": strat,
            "accuracy_%": round(acc, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
            "sec_per_item": round(avg_sec, 3),
            "tokens_per_correct": round(tok_per_correct, 1) if not math.isnan(tok_per_correct) else "N/A",
            "cost_per_1k_usd": round(c1k, 4),
            "accuracy_per_dollar": round(eff, 1) if not math.isnan(eff) else "N/A",
        })
    pd.DataFrame(rows).to_csv(f"results/cost_efficiency_seed_{seed}{sfx}.csv", index=False)
    print(f"✅ Cost-efficiency table → results/cost_efficiency_seed_{seed}{sfx}.csv")

# ── Aggregation + headline thesis table ───────────────────────────────────────

def aggregate():
    files = sorted(glob.glob("results/results_seed_*.csv"))
    if not files:
        print("No results/results_seed_*.csv found. Run the experiment first.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A model may appear in more than one file (e.g. reruns); de-dupe per row.
    df = df.drop_duplicates(subset=["model", "benchmark", "strategy", "seed"],
                            keep="last") if False else df
    print(f"Aggregating {len(files)} file(s): {files}\n")

    per_seed = (df.groupby(["model", "family", "params_b", "benchmark", "strategy", "seed"])
                  ["correct"].mean().mul(100).reset_index())
    agg = (per_seed.groupby(["model", "family", "params_b", "benchmark", "strategy"])["correct"]
             .agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "n_seeds"}))
    agg["acc_mean"] = agg["acc_mean"].round(1)
    agg["acc_std"] = agg["acc_std"].fillna(0).round(1)
    agg.to_csv("results/aggregate_accuracy.csv", index=False)
    print("── ACCURACY (mean ± std % across seeds) ──")
    print(agg.to_string(index=False))

    # GSM8K robustness table (defuses the Schaeffer 2023 metric-artifact critique:
    # show the reasoning collapse persists under a softer, less discontinuous metric).
    gg = df[df["benchmark"] == "gsm8k"]
    if len(gg):
        cont = (gg.groupby(["model", "family", "params_b", "strategy"])
                  .agg(exact_match=("correct", "mean"),
                       numeric_present=("numeric_present", "mean"),
                       median_rel_error=("rel_error", "median")).reset_index())
        cont[["exact_match", "numeric_present"]] = (cont[["exact_match", "numeric_present"]]*100).round(1)
        cont["median_rel_error"] = cont["median_rel_error"].round(3)
        cont = cont.sort_values(["family", "params_b", "strategy"])
        cont.to_csv("results/gsm8k_continuous.csv", index=False)
        print("\n\n── GSM8K ROBUSTNESS: exact-match vs numeric-present (%) ──")
        print(cont.to_string(index=False))
        print("(If numeric_present tracks exact_match and both stay low at small "
              "sizes, the reasoning collapse is NOT a metric artifact.)")

    # Headline trade-off, computed per family (sorted by parameter count).
    print("\n\n── PROMPT-vs-SCALING TRADE-OFF (per family) ──")
    rows = []
    local = agg[agg["family"] != "Gemini"]
    for family in sorted(local["family"].unique()):
        fam = local[local["family"] == family]
        for bench in sorted(fam["benchmark"].unique()):
            sub = fam[fam["benchmark"] == bench]
            sizes = list(sub.sort_values("params_b")["model"].drop_duplicates())
            zs = {m: sub[(sub.model == m) & (sub.strategy == "zero_shot")]["acc_mean"].max()
                  for m in sizes}
            best = {}
            for m in sizes:
                ms = sub[sub.model == m]
                if len(ms):
                    bi = ms.loc[ms["acc_mean"].idxmax()]
                    best[m] = (bi["strategy"], bi["acc_mean"])
            for i, m in enumerate(sizes):
                bstrat, bacc = best.get(m, ("-", float("nan")))
                beats_next = ""
                if i + 1 < len(sizes):
                    nxt = sizes[i+1]
                    beats_next = ("✓ matches/beats %s zero-shot (%.1f)" % (nxt, zs[nxt])
                                  if bacc >= zs[nxt] else
                                  "✗ below %s zero-shot (%.1f)" % (nxt, zs[nxt]))
                rows.append({"family": family, "benchmark": bench, "model": m,
                             "zero_shot": round(zs[m], 1),
                             "best_strategy": bstrat, "best_acc": round(bacc, 1),
                             "prompt_uplift_pp": round(bacc - zs[m], 1),
                             "vs_next_size_up": beats_next})
    tdf = pd.DataFrame(rows)
    tdf.to_csv("results/thesis_tradeoff.csv", index=False)
    print(tdf.to_string(index=False))
    print("\n✅ Wrote results/aggregate_accuracy.csv and results/thesis_tradeoff.csv")
    print("Read 'vs_next_size_up': ✓ = a good prompt buys ~1 model tier; "
          "✗ = scaling wins on that benchmark.")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt engineering vs. model scaling — local MLX edition.")
    parser.add_argument("--models", nargs="+",
                        default=["qwen-0.5b", "qwen-1.5b", "qwen-3b", "qwen-7b",
                                 "gemini-2.5-flash"])
    parser.add_argument("--benchmarks", nargs="+", default=["sst2", "gsm8k", "mmlu"],
                        choices=BENCHMARKS)
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=STRATEGIES,
                        help="Prompting strategies to run (default: all four).")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_cost", type=float, default=GPU_COST_PER_HOUR,
                        help="$/hr for the dollar column on local models (default 0).")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for output files so multiple runs (e.g. a "
                             "second model family) don't overwrite each other. "
                             "--aggregate merges all results_seed_*.csv regardless.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Combine all seed CSVs + emit the thesis trade-off table.")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    else:
        gemini_key = setup_env()
        print(f"\nConfiguration:")
        print(f"  Models:     {args.models}")
        print(f"  Benchmarks: {args.benchmarks}")
        print(f"  N samples:  {args.n_samples}")
        print(f"  Seed:       {args.seed}")
        print(f"  Tag:        {args.tag or '(none)'}")
        print(f"  Strategies: {args.strategies}")
        run_experiment(args.models, args.benchmarks, args.n_samples,
                       args.seed, gemini_key, args.gpu_cost, args.tag,
                       args.strategies)
