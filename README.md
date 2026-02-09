# Goal-Conditioned State Space Reasoning (GSSR)

**AGI is hiding in latent space.**

This project explores **test-time adaptation** of hybrid State Space Models (SSMs) by manipulating the recurrent latent states prior to or during during inference. The goal is to steer model behavior toward desired outcomes without any weight updates and create a lightweight form of test-time fine-tuning that leverages the natural recurrence of SSMs.

---

## Hypothesis

> I hypothesize that test-time adaptation of hybrid State Space Models (SSMs), such as those in the IBM Granite-4.0 series, can be effectively achieved through `goal-conditioned perturbations applied to recurrent latent states` during inference **OR** by `mergine cached states prior to inferernce` or some combination thereof. This approach could enable dynamic steering of model outputs toward desired behaviors without updating model parameters, thereby demonstrating a lightweight form of test-time fine-tuning that preserves training stability while enhancing adaptability to specific tasks or domains.

---

## Core Idea

### Perturbation (Steering)

1. Extract recurrent SSM states from the Granite-4.0 hybrid model (Mamba-2 + sparse attention layers) during generation.
2. Perturb selected late-layer states with a goal vector derived from prompt embeddings or refined belief states.
3. Use gating, norm clamping, and annealing to stabilize steering.
4. Observe how output trajectories shift: from coherent answers -> subtle drift -> repetition lock-in -> multilingual/code collapse at high strength.

Early results show clear steering effects but also inherent instability. Random perturbations already change behavior noticeably suggesting latent state manipulation is a viable control channel.

### State Caching

1. Extract recurrent SSM states from the Granite-4.0 hybrid model (Mamba-2 + sparse attention layers) during generation.
2. Merge pre-cached state into prompt cache using a blend ratio for strength
3. Generate response from merged cache

---

## Docs

[Hacking Granite-4.0-Hybrid](docs/dissecting_granite.md)

As a first step toward goal-conditioned latent steering (part of my research into GSSR), I hacked on IBM's Granite-4.0 hybrid model (Mamba-2 + attention), located the recurrent SSM states in the late layers, and perturbed them mid-generation with a simple random nudge. The output changed immediately from clean answers to repetition lock-in or foreign-code gibberish at higher strength.

[Building a Bayesian GSSR Goal Network](docs/building_gssr_goals.md)

Continuing the work from Hacking Granite-4.0-Hybrid (above), this write-up talks through the process of creating and conditioning the goal state to apply as perturbations to selected states in a hybrid transformer-ssm model to steer model outputs in pusuit of a viable test-time fine-tuning strategy.

[Goal-Conditioned State Caching](docs/goal_state_caching.md)

## Key Components

### Main.py

- **Model Recon**: This module breaks apart the IBM Granite-4.0-H model and identifies model layers and structure

```bash
...
Inspecting HybridMambaAttentionDynamicCache attributes...
Cache attributes:
  .conv_states: type=<class 'list'>
    length=32, first item type=<class 'torch.Tensor'>
      example shape=torch.Size([1, 1792, 4]), mean=-0.0420
  .get_mask_sizes: type=<class 'method'>
  .get_seq_length: type=<class 'method'>
  .has_previous_state: type=<class 'bool'>
  .is_compileable: type=<class 'bool'>
  .key_cache: type=<class 'list'>
    length=32, first item type=<class 'torch.Tensor'>
      example shape=torch.Size([1, 0]), mean=nan
  .layers_block_type: type=<class 'list'>
    length=32, first item type=<class 'str'>
  .reorder_cache: type=<class 'method'>
  .ssm_states: type=<class 'list'>
    length=32, first item type=<class 'torch.Tensor'>
      example shape=torch.Size([1, 48, 32, 128]), mean=0.0000
  .transformer_layers: type=<class 'list'>
    length=4, first item type=<class 'int'>
  .update: type=<class 'method'>
  .value_cache: type=<class 'list'>
    length=32, first item type=<class 'torch.Tensor'>
      example shape=torch.Size([1, 0]), mean=nan

Found 'ssm_states': type=<class 'list'>, length=32
  Mamba state 0: shape=torch.Size([1, 48, 32, 128]), mean=0.0000
  Mamba state 1: shape=torch.Size([1, 48, 32, 128]), mean=0.0000
  Mamba state 2: shape=torch.Size([1, 48, 32, 128]), mean=0.0000
  Mamba state 3: shape=torch.Size([1, 48, 32, 128]), mean=-0.0000
  Mamba state 4: shape=torch.Size([1, 48, 32, 128]), mean=0.0001
  Mamba state 5: shape=torch.Size([1, 48, 32, 128]), mean=0.0011
  Mamba state 6: shape=torch.Size([1, 48, 32, 128]), mean=0.0009
  Mamba state 7: shape=torch.Size([1, 48, 32, 128]), mean=0.0012
  Mamba state 8: shape=torch.Size([1, 48, 32, 128]), mean=0.0006
  Mamba state 9: shape=torch.Size([1, 48, 32, 128]), mean=0.0001
  Mamba state 10: shape=torch.Size([1, 0]), mean=nan
  ...
```

### Goal.py

- **Bayesian Steering Module**
- **LBS (Latent Belief System)**: A lightweight Bayesian SSM filter (GRU + variational posterior) that refines a latent state using numerical observations and semantic summaries from the LLM.
- **GoalLBS**: Adds goal encoding and steering logic that encodes target text into a latent goal vector, steers the prior belief toward it with gated proportional correction, and clamps norms to prevent divergence.
- **Perturber integration**: Captures refined states and injects them as goal vectors into Granite SSM states during inference.

### Gen.py

- **Proof of Concept** - This module demonstrates the ability to effectively reach into the model itself and apply a random perturbation to later mamba layers to affect the output of the model. Module will print `Unsteered Output` and `Steered Output`

### Test_Bench.py

- **Testing Goal Network**

```bash
Encoding Goal: 'Achieve maximum energy efficiency and system stability.'
Starting Training Step...
DEBUG tensor inputs_embeds: shape=torch.Size([1, 30, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000595
DEBUG tensor inputs_embeds after concat: shape=torch.Size([1, 38, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000496
  Step 1 | Loss: 6.2888 | KL: 1072.0000                                                                                     DEBUG tensor inputs_embeds: shape=torch.Size([1, 31, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000273
DEBUG tensor inputs_embeds after concat: shape=torch.Size([1, 39, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000313
  Step 2 | Loss: 19.7368 | KL: 1328.0000
DEBUG tensor inputs_embeds: shape=torch.Size([1, 30, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000362              DEBUG tensor inputs_embeds after concat: shape=torch.Size([1, 38, 768]), device=cuda:0, dtype=torch.bfloat16, mean=0.000370
  Step 3 | Loss: 56.1229 | KL: 1528.0000
                                                                                                                            Computing Gradients...
   Success: Gradients successfully propagated to Goal Encoder and GRU.
   Goal Encoder Grad Norm: 0.000000
   GRU Grad Norm: 1.828125
Testing Plan Generation (Latent Rollout)...
                                                                                                                            Generated Plan Sample:
------------------------------
Plan to achieve goal:
Step 1: 2
>7
) and their value for 3. 2
)
Step 2:  during that this and the process, you are using.
```

### Cache_Sandbox.py

- **Cached State Experiments** - This module enables experimentation in local generation with and without `past_key_values` (cached states).
- **Creates a LocalModel Class** with methods; `_prepare_inputs`, `summarize_and_capture_state`, `generate` and `generate_using_cache`.
- **Explores Differences in Responses**

All operations are external — no model weights are modified.

## Current Status (Feb 2026)

- Successfully located and extracted Mamba-2 recurrent states in Granite-4.0 hybrids (1B & 3B: 40 layers, 350M: 31 layers).
- Implemented step-by-step generation with real-time perturbation on late SSM states.
- Observed clear output changes: clean → semi-coherent drift → strong repetition or foreign/code tokens at higher strengths.
- Built test-time Bayesian refinement loop with KL annealing and gating.
- Gradients flow back to goal encoder and filter components.

## Next Steps & Collaboration Interest

- Refine goal encoding (prompt mean -> projection -> Bayesian smoothing -> steering).
- Achieve stable, meaningful steering (topic shift, fact injection, planning) without collapse.
- Explore multi-iteration refinement and RAG-like state aggregation.
- Test on real tasks (multi-step reasoning, long-context continuity, agentic behavior).

If you're working on SSMs, test-time adaptation, latent steering, Bayesian filtering at inference, or goal-conditioned generation, I'd love to collaborate or hear your thoughts.

Open to PRs, ideas, and/or discussions.

## How to Run

```bash
# Install deps
uv add torch transformers

# Run test/training script
uv run main.py

uv run gen.py

uv run test_bench.py

uv run train.py
```

---

### References and Recommended Reading

[LLM-Integrated Bayesian State Space Models for Multimodal Time-Series Forecasting](https://arxiv.org/pdf/2510.20952)

[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)

[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

[LaGarNet: Goal-Conditioned Recurrent State-Space Models for Pick-and-Place Garment Flattening](https://arxiv.org/abs/2508.17070)

[Act2Goal: From World Model To General Goal-conditioned Policy](https://arxiv.org/abs/2512.23541)

---

Feel free to open issues, email me at wallscreet@proton.me or DM @wallscreet on X with questions/ideas.
