# Research-Level LLM Mastery Plan

**One self-contained file. Read this section first, then follow the plan.**

---

## A. What this is

A study-and-practice roadmap for reaching **research-level mastery of LLMs**. It is not software; nothing in this repo runs.

- An **engineer** can train and ship a model using known recipes.
- A **researcher** can explain why a recipe works, say when it stops working, judge whether a paper's claim is trustworthy, and design a fair experiment on an idea of their own.

This plan builds the second. You train real (small) models from scratch, compare architectures and optimizers fairly, learn post-training, read the literature critically, and finish by testing your own hypotheses on memory, continual learning and world models.

## B. Who it is for and the limits

- **For:** one learner with a single 16GB GPU (RTX 5060 Ti, Appendix A), renting bigger GPUs only when a stage needs it.
- **What you get:** the *method* of frontier research. Claims are about trends across a scale ladder, not about frontier-scale models (§1.2).
- **No timelines.** You move on when a stage's exit criteria are met.

## C. What you can do at the end

1. Train an LLM from scratch and explain every line.
2. Explain the mechanism, risk and history of each major variant: MoE, MLA, sliding window, SSM/Mamba, Gated DeltaNet hybrids, Muon, GRPO.
3. Read a paper and state the claim, what the ablations prove, and the weakest point.
4. Run a fair experiment: tuned baselines, equal compute, seeds, a scale ladder.
5. Turn an idea into a hypothesis card, an eval defined in advance, and a result, even a negative one.

## D. How to use it

1. **Do §4.1 (readiness check)** and have Claude grade it strictly. It decides whether you need Stage 0A.
2. **Follow the route (§E) in order.** Before each stage, take its entry diagnostic (§4.2); skip topics you already answer fully.
3. **Work each concept with the cycle in §6.1:** attempt or primer → implement → read the paper → rebuild from memory → predict the ablation before running it.
4. **End every stage with the five research checks (§F),** written down, plus the stage's exit criteria and its replication unit.
5. **Keep the logs** (error, negative results, decisions, predictions, paper critiques) in §8.
6. **Use LLMs as tutor, reviewer and critic** (§7), not as a source of code you cannot explain.
7. **Treat hardware and software unknowns as operational, not as part of the roadmap.** Whether a kernel runs on your GPU, whether FP8 works, your real tokens/s and what fits in 16GB are settled by testing during Stage 0C and later stages. They never change the route or what you learn; they only decide whether a run happens locally or on rented hardware. Appendices A and B hold these notes.

## E. Route

**0 → 1 → 2 → 3 → 4A → 4B → 4C → 5 → 6 → 7 → 8 → 10, with Stage 9 at the scale gate.**

- **Start from day one, in parallel:** Track R (weekly paper critique, logs, harness) and **Stage 8 background reading**. Stage 8 is reading-only and does not depend on the earlier stages. Build the literature map and draft hypothesis cards once Stage 4C is done and baselines exist, so you can judge whether an idea is new.
- **Sequential:** Stages 0 to 3, then 4 to 7 (6 and 7 may interleave with 5).
- **Stage 10 entry gate:** Stage 3, Stage 4 (4C at minimum), Stage 8 and Track R exit criteria all met.
- **If time is tight, cut breadth first** (Stage 7 experiments, Stage 9 until needed). Never cut Track R, replication units, Stage 8 or Stage 10; they make this research-level.

## F. Five research checks (end of every stage, in writing)

1. **Mechanism:** derive or sketch the core idea from memory.
2. **Prediction:** what did I predict before the run, and was I right?
3. **Failure:** when does this stop working, and what was tried before it that failed?
4. **Weakest point:** what is the weakest point of the key paper for this stage?
5. **Scale:** would this survive 10x scale, and what would I test to find out?

## G. Engineer path vs this path

| Engineer path | This path |
|---|---|
| Implement it, make it train | Implement it, then find where it fails (L4) |
| Trust the recipe | Check the baseline was tuned and the gain survives scale |
| Skim papers | Weekly critique: claim, ablations, weakest point |
| Skip literature | Stage 8 literature map before any hypothesis |
| Stop at working models | Stage 10: hypothesis cards, scale ladder, write-ups |

## H. Done when

- You can write the training loop, AdamW, a transformer block and a chunked linear-attention step from memory.
- You have a fitted scaling curve, a defensible optimizer comparison and a data ablation, all with seeds.
- You have an equal-FLOPs table covering attention efficiency, MoE and a linear or hybrid variant, with predictions made before the runs.
- You have a post-trained small model with a measured gain and one analysed reward-hacking case.
- You have at least three paper critiques and three replication units, and any past result reproduces from a clean checkout.
- You have a literature map and at least one completed hypothesis card tested on the scale ladder, with a write-up whether it won or lost.

## I. Depth bar

Keep the depth targets in §2 and push the core to L4: transformer block, training loop, AdamW, LR schedules, experiment design and statistics. The muP transfer test (Stage 3) is required, not optional. Score each replication unit on whether you found the claim's weakest point.

---

*Everything below is the full plan. Section and stage numbers are used throughout: `Stage N` = curriculum stage, `§N` = framework section.*

## §1 Purpose and operating principles

**Goal A:** Learn to train an LLM from scratch the standard modern way, and absorb current research and architecture variants (MoE, MLA, sliding window, SSM/Mamba, Gated DeltaNet hybrids, Muon, RLVR/GRPO).
**Goal B:** Run rigorous experiments on my own architecture and intelligence hypotheses (memory, continual learning, world models).

### §1.1 Principles
1. **Attempt, then read, then rebuild.** For concepts near your level, attempt the implementation before reading the paper. For unfamiliar areas (state-space models, RL), read a short primer first.
2. **Equal-compute comparisons only.** A result without a tuned baseline and seeds is not a result.
3. **Small-scale wins often vanish at scale.** Validate across a scale ladder before believing anything.
4. **Data usually beats architecture.** Check data effects before architecture effects.
5. **Know what was tried.** For architectures and optimizers, learn: what it is, what it saves, what it risks, what was tried before, why it failed or won. Lighter treatment elsewhere.
6. **Research habits start on day one** (Track R), not at Stage 10.
7. **Depth is explicit.** Every topic has a target level (§2). Do not over-invest in L4 where L2 suffices.
8. **Skip what you already know.** Use the readiness check and each stage's entry diagnostic.

### §1.2 Scope honesty
On a 16GB card plus occasional rented GPUs you can master the *method* of frontier research: rigorous small-scale experiments, scaling curves, ablations, and critical reading. You cannot do frontier-*scale* work. Results are claims about trends across a scale ladder, not about frontier models.


---

## §2 Depth ladder and targets

| Level | Meaning |
|---|---|
| L1 | Explain it and derive the key equation |
| L2 | Implement it from memory |
| L3 | Reproduce a published result |
| L4 | Find where it fails, or extend it |

| Topic | Target |
|---|---|
| Transformer block, training loop, AdamW, LR schedules | L4 |
| Experiment design and statistics | L4 |
| Scaling laws, muP, optimizer comparison (Muon vs AdamW) | L3 |
| Linear attention, delta rule, SSM / Mamba, hybrids | L3 |
| Data filtering, dedup, mixtures | L3 |
| SFT, DPO, GRPO | L3 |
| Kernel writing (chunked linear attention or scan) | L2 |
| MLA, sliding window, MoE | L2 for all, L3 for one of your choice |
| Evals, KV cache, quantization, distillation | L2 |
| Low-precision training (bf16, FP8/FP4) | L1 to L2 |
| Distributed training | L2 |
| Interpretability, long context, test-time compute | L1 to L2 |
| Goal B literature (memory, continual learning, world models) | L1 breadth, L3 depth on the area you pick |

**Depth examples worth requiring**
- Derive the delta rule as one SGD step on the regression loss ‖Sk − v‖². This is why it fixes plain linear attention's blurry accumulation.
- Derive why muP transfers a learning rate across width, not just state that it does.
- Derive 6·N·D training FLOPs and the parameter count of one block.

---

## §3 Roadmap

| Stage | Name | Core output |
|---|---|---|
| 0 | Foundations and setup | Gaps closed, working stack, throughput numbers |
| 1 | Baseline reproduction | Faithful GPT-2 124M, then modernization ablations |
| 2 | Data and tokenization | Data ablation at 100M scale |
| 3 | Scaling, optimizers, training dynamics | Fitted scaling curve, Muon vs AdamW result |
| 4 | Architecture (4A, 4B, 4C) | Equal-FLOPs comparison table with diagnostics |
| 5 | Post-training | RL basics, then SFT, DPO, GRPO on a small model |
| 6 | Evals, inference, distillation | Quantized model served; a distilled model |
| 7 | Breadth | Interpretability, long context, test-time compute |
| 8 | Goal B literature | Literature map, what-was-tried table, candidate hypotheses |
| 9 | Scale-up and distributed | One rented multi-GPU run |
| 10 | Own hypotheses | Hypothesis cards, scale ladder, results |
| Track R | Research method and harness | Runs continuously from Stage 0 |

**Ordering rules**
- Stages 0 to 3 are sequential. Stage 3 precedes Stage 4 because tuned optimizers and muP make architecture comparisons fair.
- Stages 4 to 7 are mostly sequential; 6 and 7 may interleave with 5.
- Stage 8 comes after tooling and baselines exist, so you can judge whether an idea is new.
- **Stage 10 entry gate:** Stage 3 exit criteria, Stage 4 exit criteria (4C at minimum), **Stage 8 exit criteria**, and Track R exit criteria.

---

## §4 Readiness check and entry diagnostics

### §4.1 Readiness check (do once, before Stage 0)
Answer from memory, without looking anything up. Have Claude grade strictly.
1. Backprop through softmax → cross-entropy. What is the gradient w.r.t. the logits?
2. Parameter count of one transformer block with width d and a 4d MLP. Why is training FLOPs ≈ 6·N·D?
3. Why do we scale attention scores by 1/√d_k?
4. Why pre-norm over post-norm?
5. What does weight decay apply to in AdamW, and why not to norms and embeddings?
6. Why is decoding memory-bandwidth-bound while training is compute-bound?

**Scoring**
- 5 to 6 correct: skip the Stage 0A crash course.
- 3 to 4 correct: targeted primer on the missed items only.
- 0 to 2 correct: full Stage 0A before Stage 1.

### §4.2 Entry diagnostic (before each stage)
Answer from memory. Skip topics you answer fully.
- Stage 1: Derive shapes and FLOPs of a transformer block. Write the training loop and LR schedule unaided.
- Stage 2: Explain how near-duplicate removal works and why over-filtering hurts. Explain what BPE merges do.
- Stage 3: Explain why muP transfers hyperparameters, what Chinchilla's ratio is, and what critical batch size means.
- Stage 4A: Explain what the KV cache stores and why GQA shrinks it.
- Stage 4B: Explain top-k routing and the load-balancing problem.
- Stage 4C: Write plain linear attention as a recurrence and explain why it blurs.
- Stage 5: Write the policy gradient and the PPO clipped objective, then GRPO and how it differs from PPO.
- Stage 6: Explain per-channel vs per-group weight quantization and what speculative decoding guarantees.
- Stage 8: Name two reasons catastrophic forgetting occurs.
- Stage 9: Explain what FSDP shards and what it communicates.

Quiz prompt: *"Quiz me one question at a time, escalating. Do not reveal answers until I commit. Grade strictly."*

---

## Stage 0: Foundations and setup

**Objective:** close prerequisite gaps, learn the evaluation and statistics basics every later stage depends on, and get a trustworthy training stack.

**0A. Foundations (full version only if the readiness check says so)**
- Zero to Hero (Karpathy): micrograd and makemore. Stop before "Let's build GPT"; that becomes Stage 1.
- Matrix calculus for softmax, cross-entropy, attention, layer norm.
- Probability and information theory: entropy, cross-entropy, KL, perplexity.
- PyTorch fluency: autograd, `nn.Module`, shapes, dtypes, debugging.

**0B. Statistics and eval basics (everyone)**
- Seed noise, effect size, confidence intervals, bootstrap. When is a gain "inside noise"?
- Evals: lm-evaluation-harness, what HellaSwag and similar benchmarks can and cannot detect, contamination, why val loss is only comparable on the same data and same val split.

**0C. Stack (resolve operational unknowns here; see Appendix A)**
- Python env, CUDA 12.8+ PyTorch build, `torch.cuda.get_device_capability()` shows (12, 0).
- Config system, logging, checkpoint/resume, deterministic seeding.
- Micro-benchmarks: matmul TFLOPs, SDPA, a tiny Triton kernel, `torch.compile`, and a single flash-linear-attention kernel test.
- Record tokens/s and MFU for 124M and ~350M configs.

**Exit criteria**
- Readiness check passed (or gaps closed).
- You can state, with a number, what counts as a real effect on a benchmark of a given size.
- A tiny model overfits one batch to near-zero loss; resume reproduces the loss curve.
- Throughput numbers recorded. Token budgets for later stages come from these numbers, not guesses.

**Failure modes:** wrong PyTorch/CUDA build, silent fallback to slow kernels, non-deterministic data loading, mixed-precision bugs.

---

## Stage 1: Baseline reproduction (GPT-2 124M)

**Objective:** a baseline you fully understand, validated against a pinned reference, then an understanding of what each modern component contributes. Everything later compares to this.

**Pin one reference run before starting.** Write down: dataset, val split, tokens, tokenizer, eval method. Karpathy's llm.c reproduction on FineWeb (10B-token sample) reports validation loss near 3.29 and HellaSwag near 0.299, against GPT-2 124M's 0.294. FineWeb-Edu runs report different validation losses because data and val split differ. Never compare val loss across datasets.

**1A. Faithful GPT-2 (first)**
- Original architecture: LayerNorm, GELU, learned positions, multi-head attention, tied embeddings, padded vocab.
- GPT-3-style recipe: AdamW (0.9, 0.95), weight decay on 2D weights only, peak LR 6e-4 with warmup and cosine, ~0.5M-token batches, gradient clipping, bf16, `torch.compile`.
- Resources: Karpathy's "Let's reproduce GPT-2" and build-nanogpt; llm.c's reference dump to check your implementation before training.
- Run: the full-budget baseline is cheapest on a rented single A100/H100 (one report: ~178K tokens/s on an A100 at $1.29/hr, so about 16 hours and roughly $20 at 2024 prices; check current rates). Use local runs for debugging and short checks.

**1B. Modernization ablations (local, reduced budget)**
Swap in, one at a time: RMSNorm, RoPE, SwiGLU at matched parameters, GQA, optionally tied-embedding variants and a WSD schedule.
- **Predict first:** direction, rough size, reason. Record the prediction before running.
- **Measure seed noise first** with 3 seeds of the baseline. Then screen variants with 1 seed each; add seeds only for effects larger than about 2× the noise.
- Token budget per run comes from measured Stage 0 throughput.

**1C. Consolidation**
- Rewrite the training loop and one model component from memory.
- Derive FLOPs and parameter counts on paper.
- Start the error log and negative-results log.

**Core concepts**
- Why transformers won: RNN/LSTM vanishing gradients and sequential training; attention gave parallel training and direct long-range paths, at quadratic compute and a growing KV cache. Every Stage 4 variant targets that cost.
- Pre-norm and residual streams stabilized depth; post-norm needed delicate warmup.
- Training is compute-bound and stores activations; decoding is memory-bandwidth-bound with a KV cache.
- Optional side notes: artificial neuron vs brain; backprop's lack of locality and continual learning. This becomes central in Stage 8.

**Resources:** Karpathy (nanoGPT, build-nanogpt, nanochat); Stanford CS336 lectures (architectures, GPUs, kernels, scaling laws) as the spine for Stages 1 to 4.

**Replication unit:** reproduce the reference loss and HellaSwag curve shape; write a half-page critique of where your run differs and why.

**Exit criteria**
- Rented baseline matches the pinned reference within noise.
- The ordering of your modernization results at the reduced local budget is explained, with a prediction hit-rate recorded.
- Checkpoint resume is exact.
- Strict oral quiz passed on every line of your loop.

**Failure modes:** data-loader duplication, wrong schedule, weight decay on norms/embeddings, eval contamination, comparing against numbers from a different tokenizer or dataset.

---

## Stage 2: Data and tokenization

**Objective:** understand that data quality drives outcomes, and learn to measure it.

**Concepts (target L3)**
- Filtering, exact and near-duplicate removal, quality classifiers, mixtures, multi-phase curricula, late-stage annealing on high-quality data.
- Tokenizer size tradeoffs: embedding cost, sequence length, multilingual efficiency. Write your own BPE here.
- Synthetic data: where it helps (reasoning, textbook-like) and where it narrows diversity.
- Contamination and benchmark leakage.

**Resources:** FineWeb/FineWeb-Edu, DCLM, SmolLM2/SmolLM3 reports, OLMo 2, Hugging Face Smol Training Playbook, CS336 data lectures.

**Build:** at 100M scale, train on (a) raw, (b) filtered, (c) a mixture variant. Hold architecture fixed.

**Replication unit:** reproduce a filtering effect from FineWeb-Edu or DCLM at small scale; critique whether it would survive scale.

**Exit criteria:** you can show a data change that moves benchmarks more than a typical architecture tweak, with seeds. Cumulative review passed (§6.2).

**Failure modes:** over-filtering reduces diversity; evaluating on data derived from the training source.

---

## Stage 3: Scaling laws, optimizers, training dynamics

**Objective:** predict large-run behavior from small runs, and train stably.

**Concepts**
- **Scaling laws (L3):** Chinchilla (~20 tokens/param compute-optimal) vs modern heavy overtraining for cheaper inference. Fit your own loss-vs-compute curve; learn the fitting method, not just the result.
- **muP / hyperparameter transfer (L3):** derive why it transfers; test whether it holds for your baseline.
- **Stability toolkit:** QK-norm, z-loss, init scale, warmup length, Adam epsilon and beta2, loss-spike diagnosis.
- **Training dynamics (L1 to L2):** critical batch size, gradient noise scale, why large batches change optimizer rankings, edge-of-stability intuition. Grokking and double descent as reading.
- **Low-precision training (L1 to L2):** bf16 as the baseline; FP8 and FP4 formats; scaling factors and where they break (outliers, instability); which hardware supports what. Read, plus one small test if your card supports it; otherwise treat it as a Stage 9 item on rented hardware.
- **Optimizers (L3):**
  - AdamW is the default.
  - **Muon:** matrix-orthogonalized updates for 2D weights; AdamW still handles embeddings, norms, and the rest. Moonshot's study found Muon needs weight decay and per-parameter update-scale adjustment to scale, and reported roughly half the training FLOPs of AdamW for comparable loss (authors' claim).
  - Large-batch regime: recent work reports Muon and SOAP outperforming AdamW at very large batch sizes.
  - History: second-order optimizers (K-FAC, Shampoo) were long considered too costly or unstable; Muon and SOAP revived them with cheaper approximations.

**Build**
- LR and width sweep; fitted scaling curve across 4 model sizes.
- muP transfer test: does the best LR at small width stay best at larger width?
- Muon+AdamW hybrid vs AdamW on your baseline, tuned LRs for both, 3 seeds.

**Replication unit:** reproduce the shape of a published scaling or optimizer claim; critique whether the baseline was tuned.

**Exit criteria:** predict a larger run's final loss from smaller runs within a small error; defensible optimizer comparison; critical batch size estimated for your baseline.

**Failure modes:** untuned baseline optimizer; comparing at equal steps instead of equal tokens or FLOPs; ignoring that small-scale optimizer gains can shrink at scale.

---

## Stage 4: Architecture variants at equal FLOPs

**Objective:** feel the tradeoffs by implementing variants on your own baseline, one question at a time.

### 4A. KV-cache and attention efficiency
GQA/MQA, MLA, sliding window + global mix, gated attention.

| Variant | What it saves | What it risks | History |
|---|---|---|---|
| GQA / MQA | KV cache | Slight quality loss | MQA first, GQA is the compromise |
| MLA (DeepSeek) | KV cache via latent compression | Complexity, RoPE interaction | Used by DeepSeek, Kimi K2.5, GLM-5 |
| Sliding window + global | Memory, compute | Long-range recall | Gemma-style local/global |
| Gated attention | Training stability | Small overhead | Reported by Qwen3-Next |

### 4B. Sparsity: MoE (small scale)
Top-k routing, load balancing, expert capacity, router instability. Switch Transformer instability was addressed by routing and balancing work. Saves active FLOPs; risks memory, instability, load imbalance. Equal active FLOPs vs equal total parameters are different claims; report both.

### 4C. Sequence mixing: prerequisite chains, then implementation
Learn in this order, then implement.
1. **Linear attention** as a recurrent state; why it blurs.
2. **Gated linear attention (GLA):** decay gates.
3. **DeltaNet:** the delta rule. Derive it as one SGD step on ‖Sk − v‖².
4. **Gated DeltaNet:** `S ← α·S + β·(v − α·S·k)·kᵀ`; update memory by the prediction error with a decay gate α and write strength β.
5. **SSM line:** S4 → Mamba (selective state) → Mamba-2 and the state-space-duality link to attention.
6. **Hybrids:** a few full-attention layers among many linear/SSM layers (about 3:1 in Qwen3.5-style models; Nemotron-H reports up to about 3× faster inference than similar-size Transformers with roughly 8% attention layers). MiniMax's switches among linear, full and sparse attention show these choices are not settled.

Link to Goal B: the delta rule is a fast-weight memory update. Carry that into Stage 8.

**Kernel unit (L2, required)**
Implement one sequence-mixing primitive at three levels and compare:
1. Naive recurrent reference in PyTorch (correctness oracle).
2. Chunked / parallel form in PyTorch (chunked linear attention, or a selective scan).
3. Optionally a Triton version of the chunked form.
Check numerical agreement with the reference (including backward), then measure tokens/s and memory. Goal: understand why the chunked form exists and what bounds its speed, so novel architectures are not blocked by someone else's kernels.

**Failure history (read these)**
- Reformer, Performer, Linformer: most "efficient attention" lost to exact attention plus FlashAttention-style IO optimization.
- *Do Transformer Modifications Transfer Across Implementations and Applications?*: most tweaks do not transfer.
- Early linear attention had weak recall, leading to the delta rule and hybrids.

**Build:** one results table: loss vs FLOPs vs memory vs tokens/s, 3 seeds, plus recall diagnostics (MQAR, copying, needle). If library kernels do not run on your card, use your Kernel unit implementations or rent for kernel-dependent runs.

**Replication unit (pick one):** reproduce a small-scale MQAR gap between linear attention, delta rule, and attention; critique the claim.

**Exit criteria:** predict, before running, which variant wins on which diagnostic and explain why. 4A, 4B and 4C each produce comparison rows; the final table combines them. 4C requires a Kernel unit that matches its reference.

---

## Stage 5: Post-training

**Objective:** understand SFT, preference optimization, and RL with verifiable rewards on a small model.

**5A. RL fundamentals (before GRPO; short primer)**
- Policy gradient and REINFORCE; baselines and advantages; GAE.
- PPO: the clipped objective, value function, KL control to a reference policy.
- Why RL on language models is different: huge action space, sparse reward, reward hacking.

**5B. Concepts (target L3)**
- SFT data format, loss masking, chat templates.
- DPO: no reward model, simpler, sensitive to data quality; length bias and likelihood displacement.
- RLVR/GRPO: group-relative advantages, no critic; reward hacking, KL control, format rewards.
- Why RL helped reasoning (verifiable signal, longer chains) and where it plateaus.
- Reasoning/"thinking" modes (for example SmolLM3 dual mode) as a design pattern.

**Resources:** DeepSeek-R1 report, Tulu 3, TRL library, CS336 post-training lectures.

**Build (on a ~0.5B model, LoRA to fit 16GB):** SFT, then DPO, then GRPO on a verifiable task (math or code with unit tests). Beyond about 1B parameters, rent GPUs.

**Replication unit:** reproduce a small-scale GRPO gain on a verifiable task; critique what the reward actually measures.

**Exit criteria:** measurable gain on a held-out verifiable task, with analysis of at least one reward-hacking case. You can write the policy gradient, PPO, and GRPO objectives and state the differences.

---

## Stage 6: Evals, inference, distillation

**Objective:** serve and measure models, and transfer capability to small models.

- **Evals (deepening Stage 0B):** your own held-out evals, benchmark saturation, evaluation noise at small sizes.
- **Inference:** KV cache from scratch; why decoding is memory-bandwidth-bound; speculative decoding with a small draft model; serve via llama.cpp or vLLM and compare to your own implementation.
- **Quantization:** weight-only INT4/INT8 (GPTQ, AWQ, GGUF), KV-cache quantization, quality-loss measurement.
- **Distillation (L2):** logit vs data distillation; why small models trained on teacher outputs get disproportionate gains; where it breaks (capacity, diversity).

**Exit criteria:** a quantized model served locally with a measured quality/speed table; one distilled small model compared against a same-compute from-scratch baseline.

---

## Stage 7: Breadth (read, then one mini-experiment each)

**Objective:** knowledge of areas frontier work assumes. Targets L1 to L2.

- **Interpretability:** induction heads, superposition, probing, activation patching, sparse autoencoders. Mini-experiment: find induction heads in your Stage 1 model and relate them to your MQAR results.
- **Long context:** RoPE scaling and extension (for example YaRN-style), why needle-in-haystack overstates ability, long-context eval pitfalls. Mini-experiment: extend your Stage 1 model's context and measure degradation.
- **Test-time compute:** best-of-N, self-consistency, long chain-of-thought, budget control. Mini-experiment: compute-matched sampling vs a bigger model on a verifiable task.
- **Optional reading only:** diffusion language models, byte-level models, looped/recurrent-depth models, multimodality, safety evaluation.

**Exit criteria:** three short write-ups (claim, evidence, weakest point) and three small result tables.

---

## Stage 8: Goal B literature (memory, continual learning, world models)

**Objective:** know what has been tried and why it failed before proposing anything.

**Cover each area as:** problem, approaches, evidence, failure modes, open questions.
- **Continual learning and forgetting:** why forgetting occurs; replay, regularization (EWC-style), parameter isolation, adapter/LoRA-based approaches; why benchmarks often mislead.
- **Memory:** retrieval-augmented systems, memory layers, fast-weight programmers and their link to linear attention and the delta rule, test-time training and recent memory-augmented architectures (Titans-style). Search for newer work before relying on this list.
- **World models and prediction:** JEPA-style prediction in representation space, latent-state models, grounding.
- **Sample efficiency and verification:** what current evidence says about learning from less data; reliable verification of model outputs.

**Build**
- A **literature map**: for each area, a table of approach, claimed gain, how it was evaluated, and why it did or did not stick.
- **Define the eval before the model.** Continual learning: retention across sequential tasks, forward and backward transfer. Memory: recall under interference, not LM loss. World models: grounded prediction accuracy. LM loss will not show these gains.
- Three **candidate hypothesis cards** (template in Stage 10A), each citing the literature map.

**Exit criteria:** for each area, you can explain the two most likely reasons a new idea would fail. Three candidate cards drafted.

---

## Stage 9: Scale-up and distributed training

**Objective:** be ready to turn a validated idea into a larger run.

**Concepts (L2):** data/tensor/pipeline/sequence parallelism, FSDP/ZeRO, activation checkpointing, communication overhead, large-batch behavior, checkpointing and failure recovery, FP8/FP4 training on hardware that supports it. Muon needs layer-wise distributed handling because orthogonalization requires the full matrix.

**Resources:** Hugging Face Ultrascale Playbook, Smol Training Playbook, Megatron-LM docs, CS336 parallelism lectures.

**Build:** one rented multi-GPU run (small model) practicing FSDP and checkpoint/resume, with cost tracking.

**Exit criteria:** you can estimate cost and time for a target run and identify the likely bottleneck.

---

## Stage 10: Your hypotheses (research method)

**Entry gate:** Stage 3, Stage 4 (4C at minimum), **Stage 8**, and Track R exit criteria met.

**Before anything: should this be trained at all?** Is it novel, or does an existing model already cover it? Check against your Stage 8 literature map.

**10A. Hypothesis card (mandatory)**
Claim, mechanism, numeric prediction, kill criterion, baseline, estimated cost, **prior work and why it fell short**, **the eval defined before building**.

**10B. Locate the idea:** data, optimizer, architecture, objective, or post-training. Test at the matching level.

**10C. Diagnostics before loss:** associative recall (MQAR), copying, needle retrieval, state tracking (for example parity), in-context learning, and the Goal B evals from Stage 8.

**10D. Scale ladder:** about 10M, 50M, 150M, 500M. Fit curves for baseline and variant. Claim a win only if curves do not converge or cross when extrapolated.

**10E. Fair-comparison checklist:** tuned baselines, equal compute, equal data order, 3+ seeds, annealed final checkpoints, downstream evals.

**10F. Survival checks:** does the gain persist after LR annealing, on downstream tasks, and against a stronger baseline?

**10G. Scale gate:** rent GPUs only for ideas that pass three rungs.

**10H. LLM-agent ablation loops:** let agents run sweeps and propose variants, but keep a held-out eval they cannot see to prevent metric gaming. A human reviews every claimed win.

**10I. Write-up:** each completed hypothesis gets a short report (claim, setup, curves, negative controls, limitations), even when negative.

---

## Track R: Research method and harness (continuous from Stage 0)

**Weekly habits**
- One paper critique: core claim, what the ablations prove, weakest point.
- Update the error log, negative-results log, decision log, and prediction log.

**Harness (grows with each stage)**
- Fixed baselines checked into the repo; configs fully specify runs.
- At least 3 seeds for any claim; report effect size relative to seed noise. A gain inside noise is not a result.
- At least 3 model sizes per claim; fit curves rather than comparing single points.
- Equal comparisons: FLOPs, parameters, data, and memory/latency, reported separately.
- Same data order; tuned LR for each arm.
- Automated sweep runner and results database.

**Exit criteria (gate for Stage 10):** any past result can be reproduced from a clean checkout; at least three completed paper critiques and three replication units.

---

## §5 Spine reading (assigned per stage)

Read each item *after* you have built the relevant component.

| Stage | Reading |
|---|---|
| 1 | GPT-2 / GPT-3 papers (recipes), Llama 3 technical report (architecture and training sections) |
| 2 | FineWeb / FineWeb-Edu, DCLM, OLMo 2, SmolLM3 and the Smol Training Playbook |
| 3 | Chinchilla, muP, Moonshot "Muon is Scalable for LLM Training"; low-precision training reports |
| 4A, 4B | DeepSeek-V3 / V3.2 reports (MLA, MoE), Qwen3 materials |
| 4C | Gated DeltaNet and Kimi Linear papers, Mamba and Mamba-2, Nemotron-H and Falcon-H1 (hybrid evidence), Qwen3-Next / Qwen3.5 |
| 5 | DeepSeek-R1, Tulu 3 |
| 7 | Induction-heads and superposition work, long-context extension papers |
| 8 | Continual-learning surveys, test-time-training and memory-architecture papers, JEPA papers |
| 9 | Hugging Face Ultrascale Playbook, Megatron-LM docs |

Re-run a search for newer releases at the start of Stage 4 and Stage 8. This field changes monthly.

---

## §6 Learning method

### §6.1 Per-concept cycle
1. Primer if unfamiliar (short), or attempt first if near your level.
2. Implement.
3. Read the paper; compare with your implementation.
4. Rebuild the core from memory.
5. Predict an ablation's effect before running it; record whether you were right.

### §6.2 Cumulative review (interleaving)
- Each stage gate includes at least 5 questions from earlier stages.
- Spacing is by stage: the previous stage's items at the start of the next stage; items from two stages back at the middle of the current stage; older items at each phase boundary (after Stages 3, 6, 8).
- Turn missed items into flashcards (Anki or similar) and review them before starting later stages.

### §6.3 Depth checks
- L3/L4 topics: pass a from-memory derivation or sketch, plus a reproduction.
- L2 topics: pass a from-memory implementation.
- L1 topics: pass a short written explanation graded by Claude with the weakest-point question.

### §6.4 Research habits
- Weekly paper critique (Track R).
- One replication unit per stage, scored on whether you identified the claim's weakest point.
- Error log: every bug, root cause, fix.

### §6.5 Per-stage review loop
Re-derive one concept from memory, re-run one old experiment to confirm reproducibility, update the error log, prune what you no longer need, update this document.

---

## §7 Working with LLMs (Claude, ChatGPT, Claude Code)

| Role | Prompt pattern |
|---|---|
| Tutor | "Quiz me on X, one question at a time, escalating. Don't reveal answers until I commit." |
| Paper explainer | "State the core claim, what the ablations actually prove, and the weakest point." |
| Reviewer | "Find bugs and silent numerical issues in this training loop. Check masking, dtype, and scaling." |
| Implementation partner (Claude Code) | Let it scaffold boilerplate, tooling, and sweeps. Write the model core, loss, and the Stage 4C kernel unit yourself. Require tests against a reference implementation. |
| Critic | "Argue against my hypothesis. What confound explains this result?" |

**Over-reliance guards**
- Never accept code you cannot explain line by line.
- Re-implement each core component once from scratch without assistance.
- Models can be outdated or wrong about recent papers: check any claim that drives a decision against the primary source.
- Have a second model critique plans and results.

---

## Appendix A: Setup, hardware and cost (operational notes, not part of the roadmap)

> **Read this first.** Appendices A and B are working notes about one machine. They are **not** curriculum. Open questions here are answered by testing while you learn (mostly Stage 0C), and they only affect *where* a run happens (local vs rented) and its token budget. If a test fails, rent or substitute your own implementation and keep going; the roadmap does not change.

| Item | Spec |
|---|---|
| CPU | Ryzen 9900X |
| RAM | 32GB DDR5 |
| GPU | RTX 5060 Ti 16GB (Blackwell, sm_120) |
| Storage | 2TB Gen4 NVMe |
| Tools | ChatGPT, Claude, Claude Code |

**Principle:** measure tokens/s first, then plan token budgets. Do not guess throughput. Train fewer tokens locally and rent for full-budget reproductions.

**Resolve by testing in Stage 0** (see Appendix B for what is already known)
- flash-linear-attention kernels (Gated DeltaNet, Mamba-2) on sm_120.
- bitsandbytes, GPTQ/AWQ kernels on sm_120.
- FP8 (and FP4) training support on your card.
- Actual tokens/s for 124M and 350M models.
- 16GB limits for GRPO with LoRA at about 0.5B.

| Task | Local (5060 Ti 16GB) | Rent |
|---|---|---|
| 124M to 350M training | Yes, budgets from measured speed | Full-token reproductions |
| Architecture ablations at 100M | Yes | Multi-seed sweeps if slow |
| 500M training | Short runs | Full runs |
| 1B+ training | No | Yes |
| LoRA SFT/DPO up to about 1B | Yes | Larger models |
| GRPO up to about 0.5B with LoRA | Likely | Beyond that |
| Kernel-dependent hybrid runs | Maybe | If sm_120 kernels fail |
| Distributed practice | No | Yes |

---

## Appendix B: Verification status (operational notes, as of Oct 22, 2026)

> **Operational, not roadmap.** This records what was checked and what is still unknown about the hardware and software. Unchecked items are not blockers or gaps in the plan: you resolve them by running the tests in Stage 0C, then update this list. Re-check version-sensitive facts before relying on them.

**Confirmed this pass (web sources)**
- PyTorch has supported Blackwell since 2.7 built with CUDA 12.8.
- FlashAttention-4 does not run on desktop Blackwell (sm_120), per a March 2026 investigation on an RTX 5090; FA2 via Triton is the best available attention kernel there. This is a single investigation.
- llm.c GPT-2 124M reproduction on FineWeb (~9.89B tokens): validation loss near 3.29, HellaSwag near 0.299. GPT-2 124M HellaSwag: 0.294.
- A single A100 40GB run reported ~178K tokens/s at $1.29/hr (2024), and the full reproduction on 8×H100 reported at about 90 minutes and $20.
- One RTX 4090 reproduction reported ~135K tokens/s on FineWeb-Edu. Single report; other reports vary.
- Qwen3.5-397B-A17B (released Feb 2026): 397B total, 17B active, Gated DeltaNet and full attention at about 3:1.
- Stanford CS336 Spring 2026 offering exists, with lectures covering architectures, attention alternatives, GPUs, kernels/Triton, parallelism, scaling laws, inference, data, and RLVR post-training.

**Carried over from v1 (verified then, not re-checked this pass)**
- Muon scaling claims (authors' results), Megatron/NVIDIA Muon support, Muon+AdamW split.
- Smol Training Playbook, Ultrascale Playbook, nanochat existence and basic specs.
- Hybrid architecture evidence (Nemotron-H, Falcon-H1, Kimi Linear, MiniMax's shifts).

**Not verified; resolve by testing or search**
- flash-linear-attention kernels on sm_120.
- bitsandbytes, GPTQ/AWQ on sm_120.
- FP8/FP4 training support on your card.
- Your card's real tokens/s (the "several times slower than a 4090" figure is an estimate only).
- GRPO memory limits at 16GB.
- Current TRL/vLLM versions.

---

## §8 Living sections (update as we go)

### Progress log
| Stage | Status | Date started | Date done | Notes |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |

### Decisions made
- (for example pinned Stage 1 reference run, tokenizer, schedule, dataset subset)

### Error log
| Bug | Cause | Fix | Lesson |
|---|---|---|---|

### Negative results
| Idea | Setup | Result | Why it probably failed |
|---|---|---|---|

### Paper critiques and replication units
| Date | Paper | Claim | Weakest point | Replicated? |
|---|---|---|---|---|

### Prediction log
| Date | Experiment | Prediction | Outcome | Right? |
|---|---|---|---|---|

### Open questions
- 

### Hypothesis cards
- (copy template from Stage 10A)

---

## §9 First three actions

1. Do the readiness check (§4.1) and have Claude grade it strictly.
2. Close gaps per the scoring (Stage 0A as needed), then do Stage 0B (statistics and eval basics).
3. Run the Stage 0C stack and micro-benchmarks, record tokens/s, then pin the Stage 1 reference run.
