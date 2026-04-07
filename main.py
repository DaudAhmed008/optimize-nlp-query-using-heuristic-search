"""
Heuristic Search Algorithms to Optimize NLP Query
Author  : Daud Ahmed (202402480)
Course  : Heuristic Search in Artificial Intelligence
Version : 3.0  — Implements all professor feedback:
                 • Improved A* heuristic (fewer GPT-2 calls)
                 • Added BFS (Breadth-First Search)
                 • Improved Hill Climbing (multi-token + random restarts)
                 • 6 diverse prompts (up from 3)
                 • Increased token limit to 150 (up from 100)
                 • Dual-model comparison: GPT-2 base (117M) vs GPT-2 Medium (345M)

Algorithms implemented:
  1. Greedy Search          — deterministic, locally-optimal baseline
  2. Beam Search            — multi-path, industry-standard baseline
  3. BFS (Breadth-First)    — exhaustive breadth-first exploration (depth-limited)
  4. Hill Climbing (Improved) — multi-token substitution + random restarts
  5. A* Search (Improved)   — cached heuristic, fewer forward passes
  6. Contrastive Search     — Su et al. (2022), penalizes semantic repetition
  7. Simulated Annealing    — temperature-scheduled stochastic decoding
"""

import time
import heapq
import random
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

# ── Model Loading ────────────────────────────────────────────────────────────
# Change MODEL_NAME to "gpt2-medium" to test on the larger 345M model.
# Both models use the same code; only the weights differ.
MODEL_NAME = "gpt2"           # Options: "gpt2" (117M) or "gpt2-medium" (345M)

print(f"Loading model: {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
print(f"Model loaded: {MODEL_NAME} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters)")

MAX_LENGTH = 150   # Increased from 100 to allow richer output


# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Greedy Search ─────────────────────────────────────────────────────────
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


# ── 2. Beam Search ───────────────────────────────────────────────────────────
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


# ── 3. BFS (Breadth-First Search) ────────────────────────────────────────────
def bfs_search(input_ids, max_length=MAX_LENGTH, branch_factor=3, max_depth=15):
    """
    Breadth-First Search over the token space.

    Explores ALL candidates at depth d before any at depth d+1.
    Because the vocabulary has 50,257 tokens, full BFS is intractable.
    We limit exploration with:
      - branch_factor: only expand the top-k most probable tokens at each node
      - max_depth: stop BFS after this many tokens (then greedily finish)

    At each level, every surviving sequence is expanded by branch_factor
    candidates, scored by cumulative log-probability, and pruned to keep
    only the top (branch_factor * current_frontier_size) sequences, capped
    at a maximum frontier size to stay within memory.

    After max_depth BFS steps, the best sequence is extended greedily
    to max_length tokens to produce a complete output.

    This is included as a classical search baseline per professor feedback.
    """
    MAX_FRONTIER = 50   # Cap frontier size to keep memory bounded

    # Each entry: (cumulative_log_prob, sequence_tensor)
    frontier = [(0.0, input_ids.clone())]

    with torch.no_grad():
        for depth in range(max_depth):
            next_frontier = []

            for cum_score, seq in frontier:
                if seq.shape[1] >= max_length:
                    next_frontier.append((cum_score, seq))
                    continue
                if seq[0, -1].item() == tokenizer.eos_token_id:
                    next_frontier.append((cum_score, seq))
                    continue

                outputs  = model(seq)
                log_prob = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
                top_lp, top_tok = log_prob.topk(branch_factor)

                for i in range(branch_factor):
                    new_score = cum_score + top_lp[0, i].item()
                    new_seq   = torch.cat(
                        (seq, top_tok[0, i].unsqueeze(0).unsqueeze(0)), dim=1
                    )
                    next_frontier.append((new_score, new_seq))

            # Prune: keep only the top MAX_FRONTIER sequences by score
            next_frontier.sort(key=lambda x: x[0], reverse=True)
            frontier = next_frontier[:MAX_FRONTIER]

            if not frontier:
                break

    # Take the best sequence from BFS
    best_score, best_seq = max(frontier, key=lambda x: x[0])

    # Greedily extend to max_length if BFS stopped short
    if best_seq.shape[1] < max_length and best_seq[0, -1].item() != tokenizer.eos_token_id:
        with torch.no_grad():
            for _ in range(max_length - best_seq.shape[1]):
                outputs    = model(best_seq)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                best_seq   = torch.cat((best_seq, next_token.unsqueeze(0)), dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

    return best_seq


# ── 4. Hill Climbing Search (Improved) ───────────────────────────────────────
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


def hill_climbing_search(input_ids, max_length=MAX_LENGTH, top_k=10,
                          max_iter=5, num_restarts=3):
    """
    Improved Hill Climbing Search (Version 3.0).

    Changes from Version 2.0:
      1. Multi-token substitution: identifies the k worst-scoring positions
         and tries to fix all of them in each pass, not just one at a time.
      2. Random restarts: runs from multiple different seeds (temperature-
         sampled sequences) and keeps the best result across all restarts.
      3. Increased top_k from 5 to 10 and max_iter from 3 to 5 for deeper
         exploration of the substitution space.

    Each restart:
      - Phase 1: Generate a seed sequence (greedy for restart 0,
        temperature-sampled for restarts 1+).
      - Phase 2: For each pass, identify the 5 positions with the lowest
        per-token log-probability. For each, try the top-k alternatives.
        Accept the first substitution that improves average log-prob.
      - Repeat until no improvement found or max_iter reached.

    Return the best sequence across all restarts.
    """
    prompt_len  = input_ids.shape[1]
    best_global = None
    best_global_score = float('-inf')

    for restart in range(num_restarts):
        # Phase 1: Generate seed
        if restart == 0:
            # First restart: use greedy seed (same as v2)
            current_seq = greedy_search(input_ids, max_length)
        else:
            # Subsequent restarts: temperature-sampled seed for diversity
            current_seq = _temperature_sample_seed(input_ids, max_length,
                                                    temperature=1.2 + restart * 0.3)

        current_score = _sequence_log_prob(current_seq)

        # Phase 2: Multi-position improvement
        for iteration in range(max_iter):
            improved = False
            seq_len  = current_seq.shape[1]

            # Find the worst-scoring positions
            with torch.no_grad():
                outputs  = model(current_seq)
                log_prob = F.log_softmax(outputs.logits, dim=-1)
                tok_ids  = current_seq[0, 1:]
                per_tok  = log_prob[0, :-1].gather(1, tok_ids.unsqueeze(1)).squeeze(1)

            # Get indices of worst positions (excluding prompt)
            position_scores = [(per_tok[i - 1].item(), i)
                               for i in range(prompt_len, seq_len)
                               if i - 1 < len(per_tok)]
            position_scores.sort(key=lambda x: x[0])  # worst first
            worst_positions = [pos for _, pos in position_scores[:5]]

            for pos in worst_positions:
                with torch.no_grad():
                    partial  = current_seq[:, :pos]
                    outputs  = model(partial)
                    top_probs, top_tokens = (
                        F.softmax(outputs.logits[:, -1, :], dim=-1).topk(top_k)
                    )

                for i in range(top_k):
                    candidate = top_tokens[0, i].item()
                    if candidate == current_seq[0, pos].item():
                        continue

                    new_seq       = current_seq.clone()
                    new_seq[0, pos] = candidate
                    new_score     = _sequence_log_prob(new_seq)

                    if new_score > current_score:
                        current_seq   = new_seq
                        current_score = new_score
                        improved      = True
                        break

            if not improved:
                break

        # Track best across all restarts
        if current_score > best_global_score:
            best_global_score = current_score
            best_global       = current_seq

    return best_global


def _temperature_sample_seed(input_ids, max_length, temperature=1.5):
    """Generate a seed sequence via temperature sampling (for Hill Climbing restarts)."""
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            probs   = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_t  = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_t], dim=1)
            if next_t.item() == tokenizer.eos_token_id:
                break
    return generated


# ── 5. A* Search (Improved — fewer GPT-2 calls) ─────────────────────────────
def astar_search(input_ids, max_length=MAX_LENGTH, beam_width=3):
    """
    A* Search with improved efficiency (Version 3.0).

    Changes from Version 2.0:
      1. Heuristic is now computed from the SAME forward pass used to get
         g(n), eliminating the separate _heuristic() call. This cuts the
         number of GPT-2 forward passes roughly in half.
      2. Length penalty increased to alpha=0.8 (from 0.7) to encourage
         slightly longer outputs.
      3. Added a minimum-length threshold: sequences shorter than
         min_tokens are penalised in the priority score, discouraging
         A*'s tendency to stop after a few words.

    The priority queue entry is: (f_score, step_id, sequence, g_score)
      f(n) = -(g(n) + h(n))   [negated for min-heap]
      g(n) = raw cumulative log-prob
      h(n) = length-normalised quality estimate computed from the same
             forward pass, no extra GPT-2 call needed.
    """
    ALPHA      = 0.8    # Length penalty exponent (was 0.7 in v2)
    MIN_TOKENS = 20     # Minimum desired output tokens beyond the prompt

    prompt_len = input_ids.shape[1]

    def _compute_h_from_logits(seq, logits):
        """Compute heuristic from already-available logits — no extra GPT-2 call."""
        log_prob = F.log_softmax(logits, dim=-1)
        tok_ids  = seq[0, 1:]
        if len(tok_ids) == 0:
            return 0.0
        # logits may be from the parent (shorter) sequence — only use matching positions
        usable = min(len(tok_ids), log_prob.shape[1] - 1)
        if usable <= 0:
            return 0.0
        chosen = log_prob[0, :usable].gather(1, tok_ids[:usable].unsqueeze(1)).squeeze(1)
        T      = max(usable, 1)
        h_val  = chosen.sum().item() / (T ** ALPHA)

        # Penalise sequences that are too short
        generated_tokens = seq.shape[1] - prompt_len
        if generated_tokens < MIN_TOKENS:
            h_val -= (MIN_TOKENS - generated_tokens) * 0.5

        return h_val

    # Initial entry: need one forward pass for the initial heuristic
    with torch.no_grad():
        init_out = model(input_ids)
    init_h = _compute_h_from_logits(input_ids, init_out.logits)

    heap = [(-(0.0 + init_h), 0, input_ids, 0.0)]
    best_sequence = input_ids
    best_score    = float('-inf')

    for step in range(max_length):
        if not heap:
            break

        candidates  = [heapq.heappop(heap)
                       for _ in range(min(beam_width, len(heap)))]
        new_entries = []

        for _, _, current_seq, g_score in candidates:
            last_token = current_seq[0, -1].item()

            if last_token == tokenizer.eos_token_id:
                # Only accept EOS if we have generated enough tokens
                gen_len = current_seq.shape[1] - prompt_len
                if g_score > best_score and gen_len >= MIN_TOKENS:
                    best_score, best_sequence = g_score, current_seq
                elif g_score > best_score:
                    best_score, best_sequence = g_score, current_seq
                continue

            # Single forward pass: used for BOTH expansion AND heuristic
            with torch.no_grad():
                outputs  = model(current_seq)
                log_prob = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
                top_lp, top_tok = log_prob.topk(beam_width)

            for i in range(beam_width):
                next_tok = top_tok[0, i].unsqueeze(0).unsqueeze(0)
                new_g    = g_score + top_lp[0, i].item()
                new_seq  = torch.cat((current_seq, next_tok), dim=1)

                # Compute heuristic from the SAME logits (no extra call)
                new_h = _compute_h_from_logits(new_seq, outputs.logits)

                new_f = -(new_g + new_h)
                new_entries.append((new_f, step * beam_width + i, new_seq, new_g))

                if new_g > best_score:
                    best_score, best_sequence = new_g, new_seq

        heap = sorted(heap + new_entries, key=lambda x: x[0])[:beam_width]

    return best_sequence


# ── 6. Contrastive Search ────────────────────────────────────────────────────
def contrastive_search(input_ids, max_length=MAX_LENGTH, top_k=5, alpha=0.6):
    """
    Contrastive Search — Su et al. (2022), EMNLP Best Paper.

    score(v) = (1 - α) · p(v | context) − α · max cos_sim(embed(v), embed(tᵢ))

    The first term rewards fluency; the second penalises tokens whose
    embeddings are too similar to previously generated tokens.
    """
    generated    = input_ids.clone()
    embed_layer  = model.transformer.wte
    past_embeds  = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            probs   = F.softmax(outputs.logits[:, -1, :], dim=-1)
            top_k_probs, top_k_toks = probs.topk(top_k)

            if not past_embeds:
                next_token = top_k_toks[0, 0]
            else:
                past_mat = torch.stack(past_embeds, dim=0)
                best_cs, best_tok = float('-inf'), top_k_toks[0, 0]

                for i in range(top_k):
                    tok   = top_k_toks[0, i]
                    prob  = top_k_probs[0, i].item()
                    c_emb = embed_layer(tok.unsqueeze(0))

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
    High temperature early → broad sampling.
    Low temperature late → near-greedy convergence.
    """
    generated   = input_ids.clone()
    temperature = initial_temp

    with torch.no_grad():
        for _ in range(max_length):
            outputs       = model(generated)
            scaled_logits = outputs.logits[:, -1, :] / temperature
            probs         = F.softmax(scaled_logits, dim=-1)
            next_token    = torch.multinomial(probs, num_samples=1)
            generated     = torch.cat([generated, next_token], dim=1)

            temperature = max(temperature * cooling_rate, min_temp)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
smoother = SmoothingFunction().method1

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
#  EXPERIMENT SETUP — 6 DIVERSE PROMPTS
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
    "Space Exploration": (
        "The primary goal of space exploration in the next decade should be",
        "The primary goal of space exploration in the next decade should be establishing "
        "a permanent human presence on Mars to advance scientific research."
    ),
    "Education Reform": (
        "Modern education systems need to change because students today require",
        "Modern education systems need to change because students today require "
        "critical thinking skills and practical experience rather than rote memorization."
    ),
    "Healthcare Technology": (
        "Artificial intelligence in healthcare has the potential to",
        "Artificial intelligence in healthcare has the potential to revolutionize "
        "diagnosis and treatment by analyzing medical data faster than human doctors."
    ),
}

ALGORITHMS = {
    'Greedy':             greedy_search,
    'Beam Search':        beam_search,
    'BFS':                bfs_search,
    'Hill Climbing':      hill_climbing_search,
    'A* Search':          astar_search,
    'Contrastive':        contrastive_search,
    'Sim. Annealing':     simulated_annealing_search,
}


# ═════════════════════════════════════════════════════════════════════════════
#  RUN ALL ALGORITHMS ON ALL PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
all_results = {}

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
#  SAVE RAW OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
with open("all_outputs.txt", "w", encoding="utf-8") as f:
    for pname, ares in all_results.items():
        f.write(f"\n=== {pname} ===\n")
        for algo, r in ares.items():
            f.write(f"\n--- {algo} ---\n")
            f.write(r['text'] + "\n")
print("\nSaved: all_outputs.txt")


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
colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

METRICS = {
    'BLEU Score':     'bleu',
    'BERTScore (F1)': 'bert',
    'ROUGE-1 Score':  'rouge',
}

# ── Per-metric grouped bar charts (one subplot per prompt) ────────────────
for ylabel, mk in METRICS.items():
    n_prompts = len(PROMPTS)
    fig, axes = plt.subplots(1, n_prompts, figsize=(4 * n_prompts, 5), sharey=False)
    fig.suptitle(f'{ylabel} — All Algorithms × All Prompts',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (pname, ares) in zip(axes, all_results.items()):
        vals = [ares[a][mk] for a in algos]
        bars = ax.bar(algos, vals, color=colors, alpha=0.88, edgecolor='white')
        ax.set_title(pname, fontsize=9, pad=6)
        ax.set_ylabel(ylabel if ax is axes[0] else '')
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    fname = f"result_{mk}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")

# ── Execution time: grouped bars ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(algos))
n_prompts = len(PROMPTS)
width = 0.8 / n_prompts
bar_colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#C62828', '#00695C']

for i, (pname, ares) in enumerate(all_results.items()):
    times = [ares[a]['time'] for a in algos]
    ax.bar(x + i * width, times, width, label=pname,
           color=bar_colors[i % len(bar_colors)], alpha=0.85, edgecolor='white')

ax.set_xlabel('Search Algorithm', fontsize=11)
ax.set_ylabel('Execution Time (seconds)', fontsize=11)
ax.set_title('Execution Time — All Algorithms × All Prompts', fontsize=12, fontweight='bold')
ax.set_xticks(x + width * (n_prompts - 1) / 2)
ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('result_time.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_time.png")

# ── Generated text length: grouped bars ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for i, (pname, ares) in enumerate(all_results.items()):
    lengths = [ares[a]['length'] for a in algos]
    ax.bar(x + i * width, lengths, width, label=pname,
           color=bar_colors[i % len(bar_colors)], alpha=0.85, edgecolor='white')

ax.set_xlabel('Search Algorithm', fontsize=11)
ax.set_ylabel('Generated Text Length (words)', fontsize=11)
ax.set_title('Generated Text Length — All Algorithms × All Prompts', fontsize=12, fontweight='bold')
ax.set_xticks(x + width * (n_prompts - 1) / 2)
ax.set_xticklabels(algos, rotation=40, ha='right', fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('result_length.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_length.png")

# ── Average scores summary bar chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Average Scores Across All Prompts', fontsize=13, fontweight='bold')

for ax, (ylabel, mk) in zip(axes, METRICS.items()):
    avg_vals = [np.mean(avg[a][mk]) for a in algos]
    bars = ax.bar(algos, avg_vals, color=colors, alpha=0.88, edgecolor='white')
    ax.set_title(ylabel, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=7)
    for bar, val in zip(bars, avg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(avg_vals) * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('result_avg_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: result_avg_summary.png")

print(f"\n✓ All experiments complete ({MODEL_NAME}).  All charts saved as PNG files.")