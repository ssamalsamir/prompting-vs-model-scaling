"""Build Appendices A-D and splice them into paper/paper.html (before the note box).
A (prompt templates), B (extraction rules), D (compute) are static; C is computed
from the per-example CSVs (Wilson CIs + Holm-adjusted McNemar)."""
import pandas as pd, math
from pathlib import Path
from scipy import stats

HOME = Path.home()

# ── stats helpers ──
def wilson(c, n, conf=0.95):
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2); p = c / n; d = 1 + z*z/n
    centre = (p + z*z/(2*n)) / d
    m = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0, (centre-m)*100), min(100, (centre+m)*100))

def mcnemar(a, b):
    bb = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    cc = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    if bb + cc == 0:
        return 1.0
    return 1 - stats.chi2.cdf((abs(bb-cc) - 1)**2 / (bb+cc), 1)

def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0]*m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

files = {"Qwen2.5-Instruct": ["results/results_seed_42.csv", "results/results_seed_43.csv"],
         "Llama-3": ["results/results_seed_42_llama.csv", "results/results_seed_43_llama.csv"]}
order = {"Qwen2.5-Instruct": ["qwen-0.5b", "qwen-1.5b", "qwen-3b", "qwen-7b"],
         "Llama-3": ["llama-1b", "llama-3b", "llama-8b"]}
strategies = ["zero_shot", "few_shot", "cot", "structured"]
slabel = {"zero_shot": "zero-shot", "few_shot": "few-shot", "cot": "CoT", "structured": "structured"}
tasks = ["sst2", "gsm8k", "mmlu"]

ci_rows, mc_rows = [], []
for fam, flist in files.items():
    d = pd.concat([pd.read_csv(HOME / x) for x in flist], ignore_index=True)
    for m in order[fam]:
        for t in tasks:
            sub = d[(d.model == m) & (d.benchmark == t)]
            vecs = {}
            for s in strategies:
                ss = sub[sub.strategy == s]; vecs[s] = ss["correct"].tolist()
                n = len(ss); c = int(ss["correct"].sum())
                if n:
                    lo, hi = wilson(c, n)
                    ci_rows.append((fam, m, t, slabel[s], n, c/n*100, lo, hi))
            pairs, labels = [], []
            for s2 in ["few_shot", "cot", "structured"]:
                if vecs.get("zero_shot") and vecs.get(s2) and len(vecs["zero_shot"]) == len(vecs[s2]):
                    pairs.append(mcnemar(vecs["zero_shot"], vecs[s2])); labels.append(s2)
            for s2, p in zip(labels, holm(pairs)):
                mc_rows.append((fam, m, t, "zero-shot vs " + slabel[s2], p, stars(p)))

# ── Appendix A: prompt templates ──
A = """
<h2>Appendix A. Prompt templates</h2>
<p>Each prompt below is wrapped in the model's chat template (<code>apply_chat_template</code>,
<code>add_generation_prompt=True</code>) before generation. <code>{...}</code> denotes the
inserted example.</p>
<h3>A.1 SST-2 (sentiment)</h3>
<pre>zero-shot:  Classify the sentiment of the following sentence as exactly
            'positive' or 'negative'.\n\nSentence: {sentence}\nSentiment:

few-shot:   Classify the sentiment as 'positive' or 'negative'.
            [3 exemplars: ("The film is a masterpiece of visual storytelling." -> positive),
             ("I fell asleep twice. Complete waste of time." -> negative),
             ("A quietly moving portrait of grief and resilience." -> positive)]
            \nSentence: {sentence}\nSentiment:

CoT:        Identify the key sentiment signals in the sentence, reason through them,
            then classify as 'positive' or 'negative'. End your response with
            'Final answer: positive' or 'Final answer: negative'.\nSentence: {sentence}\nReasoning:

structured: Classify sentiment. Respond ONLY with valid JSON:
            {"sentiment": "positive"} or {"sentiment": "negative"}.\nSentence: {sentence}\nJSON:</pre>
<h3>A.2 GSM8K (arithmetic)</h3>
<pre>zero-shot:  Solve the following math problem. Give only the final number as your
            answer.\n\nProblem: {question}\nAnswer:

few-shot:   2 worked examples shown as Problem / Answer (final number only),
            then Problem: {question}\nAnswer:

CoT:        2 worked examples shown as Problem / Reasoning / Answer, then
            "Solve ... step by step. End with 'Answer: &lt;number&gt;'."
            \nProblem: {question}\nReasoning:

structured: Solve this math problem. Respond ONLY with valid JSON:
            {"answer": "&lt;number&gt;"}.\nProblem: {question}\nJSON:</pre>
<h3>A.3 MMLU (knowledge, 4-way MC)</h3>
<pre>zero-shot:  Answer the following multiple choice question. Respond with only the
            letter A, B, C, or D.\n\nQuestion: {q}\n{A..D options}\nAnswer:

few-shot:   4 exemplars (answers spread across A/B/C/D to avoid position bias),
            then Question: {q}\n{options}\nAnswer:

CoT:        "Think through each option carefully, then end your response with
            'Final answer: A' (or B, C, D)."\nQuestion: {q}\n{options}\nReasoning:

structured: Respond ONLY with valid JSON: {"answer": "A"} (or B, C, D).
            \nQuestion: {q}\n{options}\nJSON:</pre>
"""

# ── Appendix B: extraction rules ──
B = """
<h2>Appendix B. Answer-extraction rules</h2>
<p>Greedy decoding (temperature 0); answers parsed from raw model text as follows.</p>
<ul>
<li><b>SST-2.</b> structured: parse JSON <code>sentiment</code> field. CoT: regex
<code>final answer[:\\s]+(positive|negative)</code>. Otherwise: scan the last 60 characters,
then the whole output, for "positive"/"negative".</li>
<li><b>GSM8K.</b> structured: parse JSON <code>answer</code>. Otherwise: first match of
<code>answer[:\\s]+$?(number)</code>, then a trailing <code>= $number</code>, then the last
number in the text (excluding bare years 2020-2026). Scored by numeric equality (tolerance 0.01).</li>
<li><b>MMLU.</b> structured: parse JSON <code>answer</code>. CoT: regex
<code>final answer[:\\s]+([A-D])</code>. Otherwise: a lone A-D letter on the last non-empty line,
then the last A-D letter anywhere. Scored by exact letter match.</li>
</ul>
<p>Extraction failures count as incorrect. An audit of MMLU-CoT on full (untruncated) outputs
found a 2-4% extraction-failure rate, with the reported effects unchanged (&sect;4.4).</p>
"""

# ── Appendix C: computed tables ──
C = ['<h2>Appendix C. Full per-condition results (95% Wilson CIs &amp; McNemar tests)</h2>',
     '<p>Both ladders are pooled over seeds 42 and 43 (n = 600 per condition). McNemar tests are '
     'continuity-corrected and Holm-adjusted within each model&times;task, comparing each strategy '
     'to zero-shot on the same items.</p>', '<h3>C.1 Accuracy and 95% Wilson CI</h3>',
     '<table><tr><th>Family</th><th>Model</th><th>Task</th><th>Strategy</th><th>n</th>'
     '<th>Acc (%)</th><th>95% CI</th></tr>']
for fam, m, t, s, n, acc, lo, hi in ci_rows:
    C.append(f'<tr><td>{fam}</td><td>{m}</td><td>{t}</td><td>{s}</td><td>{n}</td>'
             f'<td>{acc:.1f}</td><td>[{lo:.1f}, {hi:.1f}]</td></tr>')
C.append('</table><h3>C.2 McNemar tests vs. zero-shot (Holm-adjusted)</h3>')
C.append('<table><tr><th>Family</th><th>Model</th><th>Task</th><th>Comparison</th>'
         '<th>p (adj)</th><th></th></tr>')
for fam, m, t, comp, p, st in mc_rows:
    C.append(f'<tr><td>{fam}</td><td>{m}</td><td>{t}</td><td>{comp}</td>'
             f'<td>{p:.4f}</td><td>{st}</td></tr>')
C.append('</table><p style="font-size:.8rem;color:#555;">Significance: *** p&lt;.001, '
         '** p&lt;.01, * p&lt;.05, ns = not significant.</p>')
C = "\n".join(C)

# ── Appendix D: compute ──
D = """
<h2>Appendix D. Compute and reproducibility</h2>
<p>All experiments ran on a single Apple M4 laptop (16 GB unified memory) via MLX, one model
resident at a time. Models are the 4-bit mlx-community quantizations of Qwen2.5-Instruct
(0.5/1.5/3/7B) and Llama-3 (Llama-3.2-1B/3B, Meta-Llama-3.1-8B). Decoding is greedy
(temperature 0). Max new tokens per condition: 8 (MMLU), 16 (SST-2), 48 (GSM8K non-CoT),
256 (SST-2/MMLU CoT), 512 (GSM8K CoT). n = 300 examples per condition; MMLU is stratified
across all 57 subjects. Primary results use seed 42; a second seed (43) is reported for
variance where available. The full harness (<code>run_experiments_local.py</code>), prompt
builder, and extraction code are released; <code>--aggregate</code> reproduces all tables from
the saved per-example CSVs.</p>
"""

appendix = A + B + C + D

# ── splice into paper.html (idempotent: strip any prior appendix block first) ──
paper = (HOME / "paper" / "paper.html").read_text()
import re
paper = re.sub(r"<!--APPENDIX-START-->.*?<!--APPENDIX-END-->", "", paper, flags=re.S)
block = "<!--APPENDIX-START-->\n" + appendix + "\n<!--APPENDIX-END-->\n"
marker = '<p class="note">'
paper = paper.replace(marker, block + marker, 1)
(HOME / "paper" / "paper.html").write_text(paper)
print(f"Spliced appendices A-D into paper.html ({len(ci_rows)} CI rows, {len(mc_rows)} McNemar rows).")
