# LLM-Integrated Bayesian State-Space - GoalLBS

## `GoalLBS.__init__`: The Steering Interface

The `GoalLBS` is a wrapper. It takes our existing "World Model" (`base_lbs`) and adds a **Goal Encoder** that only looks at the target we want to achieve.

This is where we move from a passive observer (the `LBS`) to an active controller (`GoalLBS`). The goal here is **steering through latent perturbation**. Instead of just letting the model drift according to the data, we are actively injecting a goal into the transition dynamics.

| Component | Purpose | Technical Role |
| --- | --- | --- |
| **`self.lbs`** | The Base World Model. | Provides the GRU, MLPs, and the Granite backbone. |
| **`self.goal_encoder`** | **Goal Latent Mapper**. | Translates high-level LLM embeddings of a "Goal" into the specific 16-dim latent space of the filter. |
| **`self.gamma`** | **Control Gain**. | Determines how aggressively the model should prioritize the goal over the natural transition. |

### The `__init__` Breakdown

```python
class GoalLBS(nn.Module):
    def __init__(self, base_lbs: LBS, goal_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.lbs = base_lbs
        self.device = base_lbs.device
        self.latent_dim = base_lbs.gru.hidden_size # 16

        # THE GOAL ENCODER
        # This is a small MLP that bridges the model's world knowledge to our state space.
        # It takes a 1536-dim embedding and squashes it to 16-dim.
        if goal_encoder is None:
            self.goal_encoder = nn.Sequential(
                nn.Linear(base_lbs.model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
                # Tanh is used to bound the goal within the same 
                # range [-1, 1] as the latent samples.
                nn.Tanh() 
            )
        else:
            self.goal_encoder = goal_encoder
        
        # Ensure we match the precision (bfloat16) and device (cuda)
        self.goal_encoder = self.goal_encoder.to(device=self.device, dtype=self.lbs.cfg.d_type)

        # THE STEERING STRENGTH
        # High gamma = "The goal is everything, ignore reality."
        # Low gamma = "Try to achieve the goal, but stay grounded in data."
        self.gamma = 0.5

```

---

## `GoalLBS.encode_goal`: The North Star

This method is only called **once** per generation or training sequence. It sets the "North Star" for the trajectory.

| Action | Logic | Purpose |
| --- | --- | --- |
| **Semantic Extraction** | `hidden_states[-1]` | We ask Granite: "What does this goal look like?" |
| **Temporal Pooling** | `hidden[0, -1, :]` | We take the embedding of the very last token. In a causal model like Granite, the final token's hidden state contains the "summary" of the entire prompt. |
| **Latent Projection** | `self.goal_encoder` | We map that massive 1536-dim concept into a specific 16-dim vector (). |

```python
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        # PROMPT ENGINEERING
        # We prompt the model to summarize the 'future state' of the goal.
        prompt = f"Goal: {goal_text}\nSummarize the desired future state."
        input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # LLM PASS (FROZEN)
        # We don't want to update the model here; we just want its interpretation on the goal.
        with torch.no_grad():
            hidden = self.lbs.model.model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        
        # GLOBAL SUMMARIZATION
        # Take the last token's representation.
        goal_emb = hidden[0, -1, :] 
        
        # LATENT MAPPING
        # Transform the embedding into our 16-dim control space.
        x_goal = self.goal_encoder(goal_emb)
        return x_goal  # Returns a [16] vector representing the "Target State"

```

---

### Why this is different from `encode_text` in our Base LBS?

While `encode_text` happens at **every step** (representing "Now"), `encode_goal` happens at the **start** (representing "Then").

In our GSSR test script, we used a `random_tensor` for steering. Here, $x_{goal}$ replaces that randomness with a semantically meaningful direction. If the goal is "Increase stock price,"  $x_{goal}$ will be a vector that, when decoded by `mlp_val`, results in a high number.

---

## `GoalLBS.steer_prior`: The Latent Nudge

This method calculates the "correction" needed to move the current trajectory toward the goal state.

The `steer_prior` method is the **actuator** of our system. In a standard Bayesian Filter, the Prior is a passive prediction of the next state. Here, we transform it into an **active steering signal**.

We are effectively performing a "Latent Nudge." By modifying the mean of the prior distribution before it reaches the KL loss or the posterior calculation, we are hopefully forcing the model's "internal imagination" to align with our desired outcome.

| Mechanism | Purpose | Technical Logic |
| --- | --- | --- |
| **Error Vector** | Directional Heading. | `x_goal - prior_mu` defines the straight-line path from where the model *thinks* it is to where we *want* it to be. |
| **Gating** | Contextual Awareness. | The `sigmoid` gate checks the "energy" of the current state. It prevents the model from steering too hard when the state is near zero (uninitialized). |
| **Annealing** | Temporal Decay. | Steering is usually strongest at the beginning and decays. This allows the model to "land" at the goal rather than crashing through it. |
| **Norm Clamping** | Stability. | Prevents the steering signal from exploding and causing `NaN` gradients, which is common in high-precision `bfloat16` networks. |

---

### Step-by-Step Steering Logic

```python
    def steer_prior(self, prior_mu, x_goal, step=0, total_steps=5):
        # CALCULATE ANNEALED GAIN
        # We reduce the pull strength (gamma) as we approach the end of the sequence.
        # This helps the model stabilize its final state.
        gamma = self.gamma * (1 - step / total_steps)
        
        # FIND THE DIRECTION
        # Simple vector subtraction: Target - Current.
        error = x_goal - prior_mu
        
        # COMPUTE THE GATE
        # We look at the magnitude of the current prior.
        # This ensures we only apply steering when the model has a "confident" state.
        gate = torch.sigmoid(prior_mu.abs().mean(dim=-1, keepdim=True))
        
        # APPLY CORRECTION
        # New Prior = Original Guess + (Strength * Confidence * Direction)
        correction = gamma * gate * error
        steered_mu = prior_mu + correction
        
        # MAGNITUDE CLAMPING (The "Speed Limit")
        # We calculate the norm of the new vector. If it's too large (> 5.0),
        # we scale it back. This is critical for preventing the latent space 
        # from "breaking" Granite's prefix-projection later.
        norm = steered_mu.norm(dim=-1, keepdim=True)
        scale = torch.clamp(5.0 / (norm + 1e-8), max=1.0)
        steered_mu = steered_mu * scale
        
        return steered_mu

```

---

### Why this works with the KL Loss

Remember the `kl_loss(prior_mu, ...)` from our BaseLBS class?
Because we pass this **steered** version of the prior into the KL Divergence calculation, the model perceives the "Goal-shifted" state as the *expected* state. When the model then looks at the actual data ($y_t, s_t$), it will try to find an interpretation of that data that fits the Goal-shifted prior.

**It effectively "tricks" the posterior into seeing the world through the lens of the goal.**

---

## `GoalLBS.forward_step`: The Steered Transition

This method executes one temporal step of the model while applying the latent "nudge" toward the goal.

The `forward_step` in `GoalLBS` is the **Integration Layer**. While `LBS.training_step` is the orchestrator for the base model, this method is the specialized version that actually executes the steering.

| Phase | Responsibility | Tech Detail |
| --- | --- | --- |
| **Observation** | Sensory Input. | Calls `lbs.encode_text` to turn current text into . |
| **Prediction** | Temporal Dynamics. | Uses the GRU to step from  to . |
| **Steering** | Perturbation. | Injects the goal vector into the Prior distribution. |
| **Refinement** | Bayesian Update. | Updates the belief into a Posterior using the steered prior. |
| **Objective** | Loss Calculation. | Computes the distance to the goal (KL) and data (Text/Value). |

---

### Step-by-Step Logic

```python
    def forward_step(self, y_t, text_t, x_prev, h_prev, x_goal, step, total_steps, training=True):
        # SEMANTIC ENCODING
        # Granite looks at the current text and compresses it to a 16-dim observation.
        s_t = self.lbs.encode_text(text_t)

        # STATE TRANSITION (The "Mind" moves forward)
        # We pass the previous sample through the GRU.
        # h_t: [num_layers, batch, latent_dim]
        _, h_t = self.lbs.gru(x_prev.unsqueeze(1), h_prev)
        
        # GENERATE RAW PRIOR
        # Extract the top-layer hidden state and guess the next distribution.
        prior_latent = h_t[-1] 
        prior = self.lbs.mlp_prior(prior_latent)
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        # APPLY THE STEERING (Crucial Step)
        # We nudge the predicted mean toward x_goal.
        # This modification is "recorded" in the gradient graph.
        steered_mu = self.steer_prior(prior_mu, x_goal, step, total_steps)

        # GENERATE POSTERIOR (The "Fact-Check")
        # Combine: RNN Context (h_t) + Data (y_t) + Semantic Obs (s_t).
        post_in = torch.cat([prior_latent, y_t, s_t], dim=-1)
        post = self.lbs.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)

        # REPARAMETERIZATION SAMPLE
        # Sample the actual state x_t from the posterior.
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        x_t = post_mu + eps * std

        # LOSS CALCULATION
        # Note: L_kl uses 'steered_mu'! This forces the model to 
        # minimize the distance between its belief and the GOAL.
        L_val = self.lbs.value_loss(x_t, y_t.mean(dim=-1)).mean()
        L_text = self.lbs.text_loss(x_t, text_t).mean()
        L_kl = self.lbs.kl_loss(steered_mu, prior_logvar, post_mu, post_logvar, step, total_steps).mean()

        # Combine with weight coefficients
        loss = (self.lbs.cfg.alpha_val * L_val +
                self.lbs.cfg.alpha_text * L_text +
                self.lbs.cfg.alpha_kl * L_kl)

        # HAND-OFF
        # If training, we return the attached tensors so gradients flow back 
        # across time steps (BPTT).
        return (x_t if training else x_t.detach(), 
                h_t if training else h_t.detach(), 
                {'loss': loss, 'val': L_val.item(), 'text': L_text.item(), 'kl': L_kl.item()},
                steered_mu)

```

---

## `GoalLBS.generate_plan`: The Manifestation

In this phase, we are no longer looking at real-world data ( or ). Instead, the model uses its own internal "imagination" to evolve the state forward and translate each state into human-readable text.

The `generate_plan` method is the **Manifestation Phase**. This is where we stop training and let the model "dream" a sequence of actions. It takes the final steered state which now contains the "essence" of our goal, and uses it to condition the model's autoregressive generation.

This method effectively performs **Deterministic Latent Rollouts**.

| Action | Logic | Purpose |
| --- | --- | --- |
| **State-to-Prefix** | `proj_state(x)` | Turns the 16-dim "idea" into high-dimensional "soft prompts" that Granite can understand. |
| **Autoregressive Gen** | `model.generate` | Granite "reads" the latent state and "writes" the corresponding step of the plan. |
| **Internal Rollout** | `self.lbs.gru(...)` | The model moves its own state forward without any external input, essentially simulating the next moment in time. |
| **Prior Projection** | `mlp_prior` | Since we have no new observation, we use the model's best guess (the prior) as the next state. |

---

### Step-by-Step Generate Plan Logic

```python
    def generate_plan(self, x_final: torch.Tensor, steps_ahead: int = 5) -> str:
        plan = "Plan to achieve goal:\n"
        x = x_final.unsqueeze(0) # Start from the final steered state
        
        # Initialize an empty GRU hidden state for the rollout
        h = torch.zeros(self.lbs.gru.num_layers, 1, self.latent_dim, device=self.device)

        for step in range(steps_ahead):
            # TRANSLATE STATE TO SEMANTICS
            # Project our 16-dim belief into 'prefix_tokens' (e.g., 8 tokens * 1536 dim).
            prefix_emb = self.lbs.proj_state(x).view(1, self.lbs.cfg.prefix_tokens, -1)
            
            # PROMPT PREPARATION
            prompt = f"Given this belief state, generate the next step in the plan.\nDate: Step {step+1}\n"
            input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # HYBRID INPUT EMBEDDING
            # We combine our latent "Soft Prompt" with the textual "Hard Prompt".
            inputs_embeds = self.lbs.model.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)

            # TEXT GENERATION
            # Granite generates text conditioned on the latent prefix.
            output = self.lbs.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.lbs.tokenizer.eos_token_id
            )
            
            # Decode only the NEW tokens (skipping the prompt and prefix)
            step_text = self.lbs.tokenizer.decode(output[0].split(input_ids.shape[1] + self.lbs.cfg.prefix_tokens)[-1], skip_special_tokens=True)
            plan += f"Step {step+1}: {step_text}\n"

            # TEMPORAL ROLLOUT (The "Dream")
            # We move the state forward in time using only the GRU and the Prior MLP.
            # This is a 'closed-loop' simulation where the model predicts its own future.
            with torch.no_grad():
                _, h = self.lbs.gru(x.unsqueeze(0), h)
                prior = self.lbs.mlp_prior(h.squeeze(0))
                prior_mu, _ = prior.chunk(2, dim=-1)
                x = prior_mu  # Update state for the next step of the plan

```

---

### Why this is the "Payoff"

This is the moment where the **Steering** we did during training (via `steer_prior`) pays off. Because the initial `x_final` was pulled toward the goal, its transition dynamics in the rollout will naturally "drift" toward states that represent the goal's fulfillment. The model will then describe these states as actionable steps.

---

### Final Blueprint Summary

We now have a complete(?) novel architecture:

1. **`LBS`**: The world model that understands history, numbers, and text.
2. **`GoalLBS`**: The control layer that nudges the world model toward a specific target.
3. **Bayesian Filtering**: The math that ensures the "nudging" remains consistent with reality.
4. **Prefix Tuning**: The interface that allows the state space to "talk" to the Transformer.
