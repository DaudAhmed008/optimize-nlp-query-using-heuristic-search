"""
Heuristic Search Algorithms to Optimize NLP Query
Author  : Daud Ahmed (202402480)
Course  : Heuristic Search in Artificial Intelligence
Version : 2.0  — Extended with 6 algorithms and multi-prompt evaluation

Algorithms implemented:
  1. Greedy Search          — deterministic, locally-optimal baseline
  2. Beam Search            — multi-path, industry-standard baseline
  3. Hill Climbing Search   — iterative local improvement from greedy seed
  4. A* Search (Improved)  — global priority queue + length-normalized heuristic
  5. Contrastive Search     — Su et al. (2022), penalizes semantic repetition
  6. Simulated Annealing    — temperature-scheduled stochastic decoding
"""

import time
import heapq
import numpy as np
import torch
import torch.nn.functional as F
import nltk
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_scorer_lib

# ── NLTK data ────────────────────────────────────────────────────────────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── 1. Load GPT-2 ────────────────────────────────────────────────────────────
model_name = "gpt2"
tokenizer  = GPT2Tokenizer.from_pretrained(model_name)
model      = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

MAX_LENGTH = 100   # Increased from 50 to allow richer output

# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 2. Greedy Search ──────────────────────────────────────────────────────────
def greedy_search(input_ids, max_length=MAX_LENGTH):
    """
    At every step picks the single highest-probability token.
    Deterministic and fast; suffers from the local-optimum problem.
    """
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_length):
            outputs    = model(generated)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated  = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return generated


# ── 3. Beam Search ───────────────────────────────────────────────────────────
def beam_search(input_ids, max_length=MAX_LENGTH, num_beams=5):
    """
    Keeps the top `num_beams` candidate sequences alive at every step.
    No-repeat-bigram constraint prevents immediate repetition.
    Uses HuggingFace's optimised batched implementation.
    """
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    return generated


# ── 4. Hill Climbing Search ──────────────────────────────────────────────────
def _sequence_log_prob(seq):
    """Average log-probability of all tokens in a sequence (quality proxy)."""
    with torch.no_grad():
        outputs  = model(seq)
        log_prob = F.log_softmax(outputs.logits, dim=-1)
        tok_ids  = seq[0, 1:]
        if len(tok_ids) == 0:
            return 0.0
        chosen = log_prob[0, :-1].gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        return chosen.mean().item()


def hill_climbing_search(input_ids, max_length=MAX_LENGTH, top_k=5, max_iter=3):
    """
    Iterative Local Search / Hill Climbing.

    Phase 1 — Seed: generate an initial complete sequence with Greedy Search.
    Phase 2 — Climb: for every non-prompt token position, try replacing the
               current token with each of the top-k alternatives predicted by
               GPT-2 at that position. Accept the replacement if and only if
               it strictly improves the sequence's average log-probability.
    Repeat until no position can be improved (local optimum) or max_iter
    iterations exhausted.

    This is a strict hill-climbing strategy: no random restarts, no downhill
    moves.  It is guaranteed to terminate because the score is bounded above
    and strictly increases on every accepted replacement.
    """
    # Phase 1: greedy seed
    current_seq   = greedy_search(input_ids, max_length)
    current_score = _sequence_log_prob(current_seq)
    prompt_len    = input_ids.shape[1]

    for _ in range(max_iter):
        improved = False
        seq_len  = current_seq.shape[1]

        for pos in range(prompt_len, seq_len):
            with torch.no_grad():
                partial  = current_seq[:, :pos]
                outputs  = model(partial)
                top_probs, top_tokens = (
                    F.softmax(outputs.logits[:, -1, :], dim=-1).topk(top_k)
                )

            for i in range(top_k):
                candidate = top_tokens[0, i].item()
                if candidate == current_seq[0, pos].item():
                    continue                        # skip the token already there

                new_seq   = current_seq.clone()
                new_seq[0, pos] = candidate
                new_score = _sequence_log_prob(new_seq)

                if new_score > current_score:       # strict improvement only
                    current_seq   = new_seq
                    current_score = new_score
                    improved      = True
                    break                           # re-evaluate from this position

        if not improved:
            break                                   # local optimum reached

    return current_seq


# ── 5. A* Search (Improved) ──────────────────────────────────────────────────
def _heuristic(sequence_ids, length_penalty=0.7):
    """
    Length-normalised heuristic  h(n).

    Original problem: raw cumulative log-prob  g(n) strictly decreases as
    sequences grow longer (all log-probs are negative), so plain A* was
    biased toward very short outputs.

    Fix (from Google's GNMT, Wu et al. 2016):
        h(n) = Σ log p(tᵢ) / T^α
    where T is the current sequence length and α ∈ (0,1) is the length
    penalty.  Normalising by T^α makes the heuristic length-agnostic,
    allowing A* to generate full-length, high-quality sequences.
    """
    with torch.no_grad():
        outputs  = model(sequence_ids)
        log_prob = F.log_softmax(outputs.logits, dim=-1)
        tok_ids  = sequence_ids[0, 1:]
        if len(tok_ids) == 0:
            return 0.0
        chosen  = log_prob[0, :-1].gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        T       = max(len(tok_ids), 1)
        return chosen.sum().item() / (T ** length_penalty)


def astar_search(input_ids, max_length=MAX_LENGTH, beam_width=3):
    """
    A* Search with length-normalised heuristic (improved over v1).

    Priority queue entry: (f_score, step, sequence, g_score)
      f(n) = -(g(n) + h(n))   [negated because heapq is a min-heap]
      g(n) = raw cumulative log-prob of chosen tokens
      h(n) = length-normalised quality estimate (see _heuristic)

    beam_width bounds the frontier size to keep memory tractable.
    The algorithm always expands the globally most-promising candidate,
    unlike Beam Search which processes candidates strictly round-by-round.
    """
    initial_h = _heuristic(input_ids)
    heap = [(-(0.0 + initial_h), 0, input_ids, 0.0)]
    best_sequence = input_ids
    best_score    = float('-inf')

    for step in range(max_length):
        if not heap:
            break

        candidates  = [heapq.heappop(heap) for _ in range(min(beam_width, len(heap)))]
        new_entries = []

        for _, _, current_seq, g_score in candidates:
            last_token = current_seq[0, -1].item()

            if last_token == tokenizer.eos_token_id:
                if g_score > best_score:
                    best_score, best_sequence = g_score, current_seq
                continue

            with torch.no_grad():
                outputs  = model(current_seq)
                log_prob = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
                top_lp, top_tok = log_prob.topk(beam_width)

            for i in range(beam_width):
                next_tok  = top_tok[0, i].unsqueeze(0).unsqueeze(0)
                new_g     = g_score + top_lp[0, i].item()
                new_seq   = torch.cat((current_seq, next_tok), dim=1)
                new_h     = _heuristic(new_seq)
                new_f     = -(new_g + new_h)
                new_entries.append((new_f, step * beam_width + i, new_seq, new_g))
                if new_g > best_score:
                    best_score, best_sequence = new_g, new_seq

        heap = sorted(heap + new_entries, key=lambda x: x[0])[:beam_width]

    return best_sequence


# ── 6. Contrastive Search ────────────────────────────────────────────────────
def contrastive_search(input_ids, max_length=MAX_LENGTH, top_k=5, alpha=0.6):
    """
    Contrastive Search — Su et al. (2022), EMNLP Best Paper.

    At every step, instead of picking the most-probable token (Greedy)
    or sampling randomly, Contrastive Search scores each top-k candidate v as:

        score(v) = (1 - α) · p(v | context)
                 − α · max  cos_sim( embed(v), embed(tᵢ) )
                           i ∈ prev tokens

    The first term rewards model confidence (fluency).
    The second term penalises tokens whose embeddings are very similar to
    any previously generated token's embedding (anti-repetition / coherence).
    α controls the trade-off; α=0 reduces to Greedy, α=1 maximises diversity.

    This is a key advance over Greedy/Beam Search: it explicitly discourages
    the semantic repetition that causes neural text degeneration.
    """
    generated    = input_ids.clone()
    embed_layer  = model.transformer.wte      # GPT-2 word-token embedding matrix
    past_embeds  = []                          # embeddings of all generated tokens

    with torch.no_grad():
        for _ in range(max_length):
            outputs        = model(generated)
            probs          = F.softmax(outputs.logits[:, -1, :], dim=-1)
            top_k_probs, top_k_toks = probs.topk(top_k)

            if not past_embeds:
                # No history yet — fall back to greedy for the first token
                next_token = top_k_toks[0, 0]
            else:
                past_mat = torch.stack(past_embeds, dim=0)   # (T, hidden_dim)
                best_cs, best_tok = float('-inf'), top_k_toks[0, 0]

                for i in range(top_k):
                    tok   = top_k_toks[0, i]
                    prob  = top_k_probs[0, i].item()
                    c_emb = embed_layer(tok.unsqueeze(0))     # (1, hidden_dim)

                    # Cosine similarity against every past embedding
                    cos = F.cosine_similarity(
                        c_emb.unsqueeze(0).expand(past_mat.shape[0], -1, -1),
                        past_mat.unsqueeze(1),
                        dim=-1
                    ).max().item()

                    cs = (1 - alpha) * prob - alpha * cos
                    if cs > best_cs:
                        best_cs, best_tok = cs, tok

                next_token = best_tok

            past_embeds.append(embed_layer(next_token.unsqueeze(0)).squeeze(0))
            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated


# ── 7. Simulated Annealing ───────────────────────────────────────────────────
def simulated_annealing_search(input_ids, max_length=MAX_LENGTH,
                                initial_temp=2.0, cooling_rate=0.92,
                                min_temp=0.1):
    """
    Simulated Annealing — temperature-scheduled stochastic decoding.

    Analogy: slow cooling of molten metal.
      • High temperature early  → tokens sampled broadly; rare/creative
        tokens have a real chance of being selected → exploration.
      • Low temperature late    → distribution sharpens toward the
        mode; high-probability tokens are almost always chosen → exploitation.

    Cooling schedule (geometric): T_{t+1} = max(T_t · r, T_min)

    Unlike Hill Climbing (which only accepts improvements) and unlike
    pure random sampling (which never converges), Simulated Annealing
    interpolates: it starts creative and becomes increasingly greedy,
    giving the output a naturally diverse opening and a coherent ending.
    """
    generated   = input_ids.clone()
    temperature = initial_temp

    with torch.no_grad():
        for _ in range(max_length):
            outputs        = model(generated)
            scaled_logits  = outputs.logits[:, -1, :] / temperature
            probs          = F.softmax(scaled_logits, dim=-1)
            next_token     = torch.multinomial(probs, num_samples=1)
            generated      = torch.cat([generated, next_token], dim=1)

            # Geometric cooling: reduce temperature each step
            temperature = max(temperature * cooling_rate, min_temp)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
smoother = SmoothingFunction().method1   # Smoothing for short outputs

def evaluate_bleu(reference, candidate):
    ref  = nltk.word_tokenize(reference.lower())
    cand = nltk.word_tokenize(candidate.lower())
    return sentence_bleu([ref], cand, smoothing_function=smoother)

def evaluate_bertscore(reference, candidate):
    _P, _R, F1 = bert_score_fn([candidate], [reference], lang='en', verbose=False)
    return F1.item()

def evaluate_rouge(reference, candidate, rouge_type='rouge1'):
    sc = rouge_scorer_lib.RougeScorer([rouge_type], use_stemmer=True)
    return sc.score(reference, candidate)[rouge_type].fmeasure


# ═════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT SETUP — MULTIPLE DIVERSE PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
PROMPTS = {
    "AI Future": (
        "The future of artificial intelligence is",
        "The future of artificial intelligence is bright and holds many possibilities."
    ),
    "Climate Change": (
        "Climate change is one of the most pressing challenges facing humanity because",
        "Climate change is one of the most pressing challenges facing humanity because "
        "it threatens ecosystems, weather patterns, and livelihoods worldwide."
    ),
    "Human Invention": (
        "The most revolutionary invention in human history has been",
        "The most revolutionary invention in human history has been the printing press, "
        "which transformed the spread of knowledge and enabled the modern world."
    ),
}

ALGORITHMS = {
    'Greedy':         greedy_search,
    'Beam Search':    beam_search,
    'Hill Climbing':  hill_climbing_search,
    'A* Search':      astar_search,
    'Contrastive':    contrastive_search,
    'Sim. Annealing': simulated_annealing_search,
}

# ═════════════════════════════════════════════════════════════════════════════
#  RUN ALL ALGORITHMS ON ALL PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
all_results = {}   # { prompt_name : { algo_name : { metric: value, ... } } }

for prompt_name, (prompt_text, reference_text) in PROMPTS.items():
    print(f"\n{'='*65}")
    print(f"  Prompt : {prompt_name}")
    print(f"  Text   : {prompt_text}")
    print(f"{'='*65}")

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    all_results[prompt_name] = {}

    for algo_name, algo_fn in ALGORITHMS.items():
        print(f"  ▶ {algo_name:<18}", end=' ', flush=True)
        t0  = time.time()
        seq = algo_fn(input_ids)
        t1  = time.time()

        text  = tokenizer.decode(seq[0], skip_special_tokens=True)
        bleu  = evaluate_bleu(reference_text, text)
        bert  = evaluate_bertscore(reference_text, text)
        rouge = evaluate_rouge(reference_text, text)

        all_results[prompt_name][algo_name] = {
            'text': text, 'bleu': bleu, 'bert': bert,
            'rouge': rouge, 'time': t1 - t0, 'length': len(text.split())
        }
        print(f"done | {t1-t0:6.1f}s  BLEU={bleu:.4f}  BERTScore={bert:.4f}  ROUGE={rouge:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  PRINT DETAILED RESULTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("DETAILED RESULTS PER PROMPT")
print("=" * 80)

for prompt_name, algo_results in all_results.items():
    print(f"\n{'─'*65}")
    print(f"  Prompt: {prompt_name}")
    print(f"{'─'*65}")
    for algo, r in algo_results.items():
        print(f"\n  Algorithm : {algo}")
        preview   = r['text'][:140] + ('…' if len(r['text']) > 140 else '')
        print(f"  Text      : {preview}")
        print(f"  BLEU      : {r['bleu']:.4f}   BERTScore: {r['bert']:.4f}"
              f"   ROUGE: {r['rouge']:.4f}")
        print(f"  Time      : {r['time']:.2f}s   Length: {r['length']} words")


# ═════════════════════════════════════════════════════════════════════════════
#  AVERAGE SCORES ACROSS ALL PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("AVERAGE SCORES ACROSS ALL PROMPTS")
print("=" * 80)

avg = {a: {m: [] for m in ['bleu', 'bert', 'rouge', 'time', 'length']}
       for a in ALGORITHMS}
for pr in all_results.values():
    for algo, r in pr.items():
        for m in avg[algo]:
            avg[algo][m].append(r[m])

print(f"\n{'Algorithm':<20} {'BLEU':>8} {'BERTScore':>10} {'ROUGE':>8}"
      f" {'Time(s)':>9} {'Words':>7}")
print("-" * 66)
for algo in ALGORITHMS:
    b  = np.mean(avg[algo]['bleu'])
    bt = np.mean(avg[algo]['bert'])
    r  = np.mean(avg[algo]['rouge'])
    t  = np.mean(avg[algo]['time'])
    l  = np.mean(avg[algo]['length'])
    print(f"{algo:<20} {b:>8.4f} {bt:>10.4f} {r:>8.4f} {t:>9.2f} {l:>7.1f}")

max_bleu = max(np.mean(avg[a]['bleu']) for a in ALGORITHMS)
print("\nRelative Accuracy (BLEU-based, vs best algorithm):")
for algo in ALGORITHMS:
    pct = (np.mean(avg[algo]['bleu']) / max_bleu) * 100
    print(f"  {algo:<20}: {pct:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════
algos  = list(ALGORITHMS.keys())
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

METRICS = {
    'BLEU Score':    'bleu',
    'BERTScore (F1)':'bert',
    'ROUGE-1 Score': 'rouge',
}

# ── Per-metric grouped bar charts (one subplot per prompt) ────────────────
for ylabel, mk in METRICS.items():
    fig, axes = plt.subplots(1, len(PROMPTS), figsize=(18, 5), sharey=False)
    fig.suptitle(f'{ylabel} — All Algorithms × All Prompts',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (pname, ares) in zip(axes, all_results.items()):
        vals = [ares[a][mk] for a in algos]
        bars = ax.bar(algos, vals, color=colors, alpha=0.88, edgecolor='white')
        ax.set_title(pname, fontsize=10, pad=6)
        ax.set_ylabel(ylabel if ax is axes[0] else '')
        ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fname = f"result_{mk}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")

# ── Execution time: grouped bars ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(algos))
width = 0.25
bar_colors = ['#1565C0', '#2E7D32', '#E65100']

for i, (pname, ares) in enumerate(all_results.items()):
    times = [ares[a]['time'] for a in algos]
    ax.bar(x + i * width, times, width, label=pname,
           color=bar_colors[i], alpha=0.85, edgecolor='white')

ax.set_xlabel('Search Algorithm', fontsize=11)
ax.set_ylabel('Execution Time (seconds)', fontsize=11)
ax.set_title('Execution Time — All Algorithms × All Prompts', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('result_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_time.png")

# ── Generated text length: grouped bars ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for i, (pname, ares) in enumerate(all_results.items()):
    lengths = [ares[a]['length'] for a in algos]
    ax.bar(x + i * width, lengths, width, label=pname,
           color=bar_colors[i], alpha=0.85, edgecolor='white')

ax.set_xlabel('Search Algorithm', fontsize=11)
ax.set_ylabel('Generated Text Length (words)', fontsize=11)
ax.set_title('Generated Text Length — All Algorithms × All Prompts', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('result_length.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_length.png")

# ── Average scores summary bar chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Average Scores Across All Prompts', fontsize=13, fontweight='bold')

for ax, (ylabel, mk) in zip(axes, METRICS.items()):
    avg_vals = [np.mean(avg[a][mk]) for a in algos]
    bars = ax.bar(algos, avg_vals, color=colors, alpha=0.88, edgecolor='white')
    ax.set_title(ylabel, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=8)
    for bar, val in zip(bars, avg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(avg_vals) * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

plt.tight_layout()
plt.savefig('result_avg_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_avg_summary.png")

print("\n✓ All experiments complete.  All charts saved as PNG files.")

with open("all_outputs.txt", "w") as f:
    for pname, ares in all_results.items():
        f.write(f"\n=== {pname} ===\n")
        for algo, r in ares.items():
            f.write(f"\n--- {algo} ---\n")
            f.write(r['text'] + "\n")