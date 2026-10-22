# Research-Level Path: Mastering LLMs the Way a Researcher Does

Companion to `llm_mastery_plan_v3.md`. Same stages, exit criteria and depth ladder (§2). This file sets the route and the depth bar.

## What this is, in plain words

**A study-and-practice roadmap, not software.** Nothing in this repo runs.

**The target is research-level mastery, not engineer-level mastery.**
- An engineer can train and ship a model using known recipes.
- A researcher can say *why* a recipe works, *when it stops working*, whether a new paper's claim is trustworthy, and can design an experiment that tests an idea of their own.

**What you end up able to do:**
1. Train an LLM from scratch and explain every line.
2. Explain the mechanism behind each major variant (MoE, MLA, sliding window, SSM, Gated DeltaNet hybrids, Muon, GRPO), what it risks, and what was tried before it.
3. Read a paper and name the claim, what the ablations prove, and the weakest point.
4. Run a fair experiment: tuned baselines, equal compute, seeds, a scale ladder.
5. Turn an idea (memory, continual learning, world models) into a hypothesis card, a defined eval, and a result, even a negative one.

**What it is not:** a way to build a frontier-scale model. A 16GB card plus occasional rented GPUs gives you the *method* of frontier research, and claims about trends across a scale ladder, not about large models.

**How to use it:** follow the route in order. A stage is done only when its exit criteria and its research checks (below) are met. There are no dates.

## Why this differs from an engineer path
| Engineer path | This path |
|---|---|
| Implement it, make it train | Implement it, then find where it fails (L4) |
| Trust the recipe | Check if the baseline was tuned, and whether the gain survives scale |
| Skim papers | Weekly critique: claim, ablations, weakest point |
| Skip literature | Stage 8 literature map before any hypothesis |
| Stop at working models | Stage 10: hypothesis cards, scale ladder, write-ups |
| Standard-stack breadth | Breadth, but every stage ends with a replication and a critique |

## Route

0 → 1 → 2 → 3 → 4A → 4B → 4C → 5 → 6 → 7 → 8 → 10, with 9 at the scale gate.

**Start in parallel from day one:** Track R (weekly paper critiques, logs, harness) and Stage 8 background reading. Stage 8 is reading-only and does not depend on the earlier stages, so doing it alongside them saves the most time. Draft hypothesis cards once Stage 4C is done.

**Do not cut:** Track R, replication units, Stage 8, Stage 10. These are what make the path research-level. If time is tight, cut breadth first (Stage 7 experiments, Stage 9 until needed).

## Research checks added to every stage
Before leaving a stage, answer in writing:
1. **Mechanism:** derive or sketch the one core idea from memory.
2. **Prediction:** what did I predict before the run, and was I right?
3. **Failure:** under what conditions does this stop working? What was tried before it and why did that fail?
4. **Weakest point:** what is the weakest point of the key paper for this stage?
5. **Scale:** would this result survive 10x scale? What would I need to test it?

## Per-stage focus

| Stage | Research focus | Why-not material to study |
|---|---|---|
| 0 | Seed noise, effect size, eval validity: what counts as a real effect | Contamination, val loss not comparable across datasets |
| 1 | Reproduce a pinned reference; predict each modernization ablation before running | Untuned baselines, weight decay mistakes, comparing across tokenizers |
| 2 | Show a data change that beats an architecture tweak, with seeds | Over-filtering, eval on data from the training source |
| 3 | Fit a scaling curve; test muP transfer (not optional here); Muon vs AdamW with tuned LRs; critical batch size | Small-scale optimizer gains that vanish at scale |
| 4A | GQA, MLA, sliding window; one at L3 | Quality loss, RoPE interaction with MLA |
| 4B | Small MoE; report equal-active-FLOPs and equal-total-params separately | Router instability, load imbalance |
| 4C | Linear attention → GLA → DeltaNet → Gated DeltaNet → Mamba → hybrids; kernel unit; recall diagnostics (MQAR, copying, needle) | Reformer/Performer/Linformer lost to FlashAttention; most tweaks do not transfer |
| 5 | RL primer, SFT, DPO, GRPO on a small model | Reward hacking, length bias, what the reward actually measures |
| 6 | KV cache, quantization, distillation | Quantization quality loss, distillation limits |
| 7 | Induction heads, long-context extension, test-time compute | Needle-in-haystack overstating ability |
| 8 | Literature map for memory, continual learning, world models, sample efficiency; define evals before models | Why each area's past attempts failed; the two likeliest reasons a new idea fails |
| 9 | One rented multi-GPU run at the scale gate | Communication overhead, large-batch behaviour |
| 10 | Hypothesis cards, scale ladder (~10M, 50M, 150M, 500M), survival checks, write-ups | Confounds, annealing effects, stronger baselines |

## Research bar (replaces "trim" columns)
- Keep v3's depth targets, and push the core to L4: transformer block, training loop, AdamW, LR schedules, experiment design and statistics.
- Keep every replication unit. Score each one on whether you found the claim's weakest point.
- Keep the weekly paper critique and the logs (error, negative results, decisions, predictions).
- Define the eval before the model in any Goal B work. LM loss does not show memory or continual-learning gains.
- Let an LLM agent run sweeps only with a held-out eval it cannot see, and review every claimed win yourself.

## Done when
- You can write the training loop, AdamW, a transformer block and a chunked linear-attention step from memory.
- You have a fitted scaling curve, a defensible optimizer comparison and a data ablation, all with seeds.
- You have an equal-FLOPs table covering attention efficiency, MoE and a linear or hybrid variant, with predictions made before the runs.
- You have a post-trained small model with a measured gain and one analysed reward-hacking case.
- You have at least three paper critiques and three replication units, and any past result reproduces from a clean checkout.
- You have a literature map and at least one completed hypothesis card tested on the scale ladder, with a write-up whether it won or lost.
