# Goal A Path: Mastering the Standard Modern LLM Stack

Companion to `llm_mastery_plan_v3.md`. Same stage numbers, same exit criteria, same depth ladder (§2). Only the route changes.

**Goal A:** train an LLM from scratch the standard modern way and absorb current architecture and training variants (MoE, MLA, sliding window, SSM/Mamba, Gated DeltaNet hybrids, Muon, RLVR/GRPO).

## Route

0 → 1 → 2 → 3 → 4A → 4B → 4C → 5 → 6 → 7 (light) → 9

**Deferred to Goal B:** Stage 8 (memory/continual-learning literature) and Stage 10 (own hypotheses). Come back to them once Goal A is done; Stages 3, 4 and Track R are already their entry gate.

## Why this order
- **0 → 1:** nothing later is trustworthy without a baseline you understand line by line and a stats habit (seed noise).
- **2 before 3:** data moves results more than most architecture tweaks (principle 4), and scaling fits need a fixed dataset.
- **3 before 4:** tuned optimizers and a fitted scaling curve make architecture comparisons fair. Without them, Stage 4 results are noise.
- **4A → 4B → 4C:** easiest to hardest. GQA/MLA reuse the attention you wrote in Stage 1; MoE adds routing; 4C needs new math (delta rule, scans) and the kernel unit.
- **5 after 4:** post-training needs a model worth post-training, and RL needs the primer (5A) before GRPO.
- **6, 7 after 5:** inference, quantization and distillation are easier once you have a trained and post-trained model to serve.
- **9 last:** distributed training matters only after you know what you are scaling.

## Priority tiers (when time is tight, cut from the bottom)

| Tier | Content | Target depth |
|---|---|---|
| Core | Stage 1 (1A, 1B, 1C), Stage 3 scaling fit + Muon vs AdamW, Stage 4C incl. kernel unit, Stage 5 (SFT, DPO, GRPO) | L3 to L4 |
| Standard | Stage 2 (data ablation), Stage 4A, 4B, Stage 6 (KV cache, quantization, distillation) | L2 to L3 |
| Light | Stage 3 muP test, low-precision training, Stage 7 (three mini-experiments, reading-level), Stage 9 (one rented run) | L1 to L2 |

## Per-stage notes for Goal A

| Stage | Do | Trim |
|---|---|---|
| 0 | Readiness check; 0B stats; 0C stack and tokens/s. Resolve the Appendix B unknowns now (sm_120 kernels, FP8, GRPO memory). | 0A only if the readiness check says so |
| 1 | Faithful GPT-2 reproduction, then all modernization ablations with predictions logged | None; this is the foundation |
| 2 | Raw vs filtered vs mixture at 100M; write your own BPE | Skip synthetic-data depth |
| 3 | Scaling fit (4 sizes), stability toolkit, Muon vs AdamW with tuned LRs | muP transfer test optional; FP8/FP4 as reading |
| 4A | GQA, MLA, sliding window; pick one for L3 | Gated attention as reading |
| 4B | Small MoE; report both equal-active-FLOPs and equal-total-params | |
| 4C | Linear attention → GLA → DeltaNet → Gated DeltaNet → Mamba line → hybrids; kernel unit (naive, chunked, optional Triton) | Triton version optional |
| 5 | 5A RL primer, then SFT → DPO → GRPO on a ~0.5B model with LoRA | Rent beyond ~1B |
| 6 | KV cache from scratch, speculative decoding, one quantization method, one distilled model vs from-scratch baseline | |
| 7 | One mini-experiment each: induction heads, context extension, test-time compute | Optional reading list dropped |
| 9 | One rented multi-GPU FSDP run with cost tracking | |

## Track R (lighter version for Goal A)
Keep: fixed baselines in the repo, configs that fully specify runs, 3 seeds for any claim, error/prediction/negative-results logs, and one paper critique a week using the §5 spine reading. Drop for now: the sweep database and the "3 replications" gate, which only matter for Stage 10.

## Replication units (one per stage, as in v3)
Stage 1 (GPT-2 curve), 2 (a FineWeb-Edu or DCLM filtering effect), 3 (scaling or Muon claim), 4C (MQAR gap), 5 (GRPO gain). These give you the three replications Stage 10 needs later.

## Goal A done when
- You can write the training loop, AdamW, a transformer block, and a chunked linear-attention step from memory.
- You have a fitted scaling curve, a defensible Muon-vs-AdamW result, and a data ablation with seeds.
- You have an equal-FLOPs table covering GQA/MLA, MoE and a linear/hybrid variant, with predictions made before the runs.
- A post-trained small model shows a measurable gain on a held-out verifiable task, with one reward-hacking case analysed.
- A quantized model is served locally with a quality/speed table.
- One distributed run on rented GPUs is complete, and you can estimate cost and bottleneck for a larger run.
