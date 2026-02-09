# Building a Bayesian GSSR Goal State

## `LBS.__init__`: The Foundation

The goal of this initialization is to set up a **Recurrent State Space Model (RSSM)** that uses a pre-trained LLM as both an encoder (sensor) and a decoder (predictor).

| Component | Purpose | Technical Role |
| --- | --- | --- |
| **`self.model` & `self.tokenizer**` | The "world knowledge" backbone. | Used to extract semantic features from text and generate forecasts based on latent states. |
| **`self.gru`** | **Deterministic State Transition**. | Models how the state evolves over time (). It provides the temporal "memory" of the system. |
| **`self.mlp_prior`** | **Stochastic Transition Prior**. | Learns to predict the *parameters* (mean and variance) of the next state based on the GRU hidden state. This represents what the model *expects* to happen. |
| **`self.mlp_post`** | **Stochastic Posterior (The Filter)**. | The "Correction" step. It refines the prior belief using current observations ( and ). It outputs the refined mean and variance. |
| **`self.mlp_val`** | **Value Predictor**. | A regressor that maps the latent state to a scalar value (). This anchors the latent space to real-world numerical data. |
| **`self.proj_sum`** | **Observation Encoder Projection**. | Compresses the high-dimensional hidden states of the LLM (1536) into your compact latent dimension (16). |
| **`self.proj_state`** | **Latent-to-Prefix Projection**. | Maps the latent state back into the LLM's embedding space as "soft prompts" or prefix tokens to guide generation. |
| **`self.sum_ids`** | **Special "Summary" Tokens**. | These act as "bottleneck" tokens. By appending these to text, we force the LLM to aggregate the preceding information into these specific positions. |

---

### Layer-by-Layer Breakdown

```python
class LBS(nn.Module):
    def __init__(self, model_path="ibm-granite/granite-4.0-h-1b"):
        super().__init__()
        # ... [LLM Loading] ...

        # THE TEMPORAL ENGINE
        # This keeps track of history. It receives the previous latent sample (x_prev) and moves the hidden state (h_prev) forward.
        self.gru = nn.GRU(self.cfg.latent_dim, self.cfg.latent_dim,
                        num_layers=self.cfg.gru_layers, batch_first=True)
        
        # THE TRANSITION PRIOR (Distribution over the "future")
        # Takes the GRU hidden state and outputs a mu and logvar.
        # Shape: [latent_dim] -> [latent_dim * 2] (for mu and logvar)
        self.mlp_prior = nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim * 2)
        
        # THE POSTERIOR FILTER (Distribution over "now")
        # Combines: 1. RNN State (History) + 2. Numerical Obs (y_t) + 3. Semantic Obs (s_t)
        # It corrects the prior based on what we just observed.
        self.mlp_post = nn.Linear(self.cfg.latent_dim * 2 + 1, self.cfg.latent_dim * 2)
        
        # REWARD/VALUE ESTIMATOR
        # Decodes the latent state into a single float. Used for MSE loss against data.
        self.mlp_val = nn.Linear(self.cfg.latent_dim, 1)
        
        # SEMANTIC COMPRESSION HEAD
        # Maps LLM hidden_size (1536) -> latent_dim (16). 
        # This is the "eye" of the model, seeing the LLM's interpretation of text.
        self.proj_sum = nn.Linear(self.model.config.hidden_size, self.cfg.latent_dim)
        
        # SEMANTIC GENERATION HEAD
        # Maps latent_dim (16) -> (Prefix_Tokens * hidden_size).
        # This allows the latent state to "speak" back to the LLM.
        self.proj_state = nn.Linear(self.cfg.latent_dim, self.cfg.prefix_tokens * self.model.config.hidden_size)

        # INFORMATION BOTTLE-NECK TOKENS
        # Pre-calculated IDs for <SUM0>, <SUM1>... to extract concentrated embeddings.
        self.sum_ids = torch.tensor(...)

```

---

## `LBS.encode_text`: The Semantic Bottleneck

This method transforms a high-level text description into a compact semantic vector () that the Bayesian filter can digest. In this step, we are bridging the gap between raw text and our latent space. This is the **Observation Model** (or "Encoder") of our system.

The `encode_text` method isn't just a simple embedding, it’s a **semantic bottleneck**. We are forcing the entire model to compress an arbitrary length string into a fixed set of "summary vectors."

| Action | Purpose | Detail |
| --- | --- | --- |
| **Prompt Wrapping** | Contextual Grounding. | Tells the LLM that the following text is "Information to be encoded." |
| **Token Concatenation** | Bottleneck Injection. | We append `sum_ids` () to the end of the input. These act as "buckets" for the LLM's attention mechanism to dump information into. |
| **Hidden State Extraction** | Feature Mining. | We grab the *last* hidden layer. These vectors contain the richest semantic representation after the Transformer blocks have processed the prompt. |
| **Projection & Pooling** | Dimensionality Reduction. | We map the LLM's 1536-dim space down to 16-dim via `proj_sum`, then average them to create a single global semantic observation. |

---

### Step-by-Step Encode Logic

```python
    def encode_text(self, text: str) -> torch.Tensor:
        # TEMPLATE THE INPUT
        # We give the LLM a clear "instruction" to behave like an encoder.
        prompt = f"Information encoded into a sequence of vectors.\n{text}\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # APPEND THE BOTTLE-NECK TOKENS
        # We add the <SUM> tokens to the sequence. 
        # Because Transformers are causal (or use attention masks), these tokens will "look back" at the text to build their representation.
        input_ids = torch.cat([input_ids, self.sum_ids.unsqueeze(0)], dim=1)

        # FORWARD PASS (NO GRADIENTS)
        # We use torch.no_grad() because the LLM backbone is frozen here and we only want to train the projection head (proj_sum).
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        
        # EXTRACT THE "SUMMARY" STATES
        # hidden_states[-1] has shape: [Batch, Seq_Len, 1536]
        # We slice the last 'sum_tokens' (the vectors corresponding to our <SUM> IDs).
        hidden = out.hidden_states[-1]
        sum_hidden = hidden[0, -self.cfg.sum_tokens:] # [sum_tokens, 1536]

        # PROJECT TO LATENT SPACE
        # Map 1536 -> 16.
        s = self.proj_sum(sum_hidden) # [sum_tokens, 16]
        
        # MEAN POOLING
        # Collapse the summary tokens into one 16-dimensional semantic observation.
        s = s.mean(dim=0, keepdim=True) # [1, 16]
        return s

```

### Why this matters for Steering

By using `<SUM>` tokens, we are creating a stable "interface" for our state space. If we just used the last token of the sentence, our latent vector would be heavily biased by the final punctuation or word. By averaging across multiple summary tokens, we get a more robust "gist" of the information.

---

## `LBS.filter_step`: The Bayesian Transition

The `filter_step` is the conceptual "engine room" of our Bayesian State Space model. This is where the **Predict-Correct** cycle happens. It manages the transition from the past belief into a new, updated belief based on incoming data.

In control theory terms, this is effectively a **Learnable Kalman Filter** wrapped in a **Variational Autoencoder (VAE)**.

This method performs two critical roles: it generates a **Prior** (guessing the next state based on history) and a **Posterior** (refining that guess with actual evidence).

| Phase | Mathematical Logic | Goal |
| --- | --- | --- |
| **Prediction (The Prior)** | $p(x_t | x_{t-1}, h_{t-1})$ |
| **Refinement (The Posterior)** | $q(x_t | h_t, y_t, s_t)$ |
| **Reparameterization** | $x_t = \mu + \epsilon \cdot \sigma$ | Allows the model to remain differentiable while sampling from a distribution. |

---

### Step-by-Step Filter Logic

```python
    def filter_step(self, x_prev, h_prev, y_t, s_t):
        # SHAPE STABILIZATION
        # Ensuring inputs are tensors with batch dimensions [1, dim] to prevent broadcasting errors during concatenation.
        if y_t.dim() == 0: y_t = y_t.unsqueeze(0)
        if s_t.dim() == 1: s_t = s_t.unsqueeze(0)

        # DETERMINISTIC TRANSITION (RNN)
        # We pass the previous latent sample (x_prev) into the GRU. h_t now contains the "temporal context" updated with the latest step.
        gru_in = x_prev.unsqueeze(1) # Add sequence dimension [Batch, 1, 16]
        _, h_t = self.gru(gru_in, h_prev)
        
        # GENERATE THE PRIOR (The "Guess")
        # We use the GRU's current memory to predict the distribution of the next state.
        # We split (or chunk) the output into Mean (mu) and Log-Variance (logvar).
        prior = self.mlp_prior(h_t.squeeze(1))
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        # GENERATE THE POSTERIOR (The "Fact-Check")
        # We concatenate history (h_t), numerical data (y_t), and semantic data (s_t).
        # This MLP 'corrects' the prior based on these external observations.
        post_in = torch.cat([h_t.squeeze(1), y_t.unsqueeze(1), s_t], dim=-1)
        post = self.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)

        # THE REPARAMETERIZATION TRICK
        # We sample a specific latent vector (x_t) from the posterior distribution.
        # Using 'eps' (random noise) ensures we can still backpropagate through the mean/std.
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std, dtype=self.cfg.d_type)
        x_t = post_mu + eps * std

        return x_t, (prior_mu, prior_logvar), (post_mu, post_logvar), h_t

```

### Why this is the "Heart"

This method is where our **Steering** will eventually live (inside `GoalLBS`). By calculating a `prior_mu` (what the model thinks will happen), we can "nudge" that mean toward our Goal vector before it ever sees the real data. If our steering is strong, the model will essentially try to "hallucinate" evidence that supports our goal.

---

Moving from the **Internal State** (the latent vector ) back out to the **External World**. The following two loss functions define the "meaning" of our latent space. These loss functions force the latent space to represent something useful and without them, would just be random noise.

---

## `LBS.value_loss`: The Grounding Mechanism

This is the simplest part of the network, but it acts as the "anchor." It forces the 16-dimensional vector to contain enough information to reconstruct the numerical data point ().

| Component | Logic | Purpose |
| --- | --- | --- |
| **`mlp_val`** | Latent  Scalar | Maps the complex latent vector back into a single number. |
| **MSE Loss** | Mean Squared Error | Ensures that if our data says "Stock price is 150," the latent state actually encodes the "150-ness." |

```python
    def value_loss(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        # DECODE THE LATENT
        # We try to predict the numerical observation (y_t) from the sampled state (x_t).
        y_pred = self.mlp_val(x_t)
        
        # CALCULATE ERROR
        # Standard Mean Squared Error. This forces the latent space to be "grounded" in the actual numerical magnitude of the data.
        return F.mse_loss(y_pred.squeeze(-1), y_t)

```

---

## `LBS.text_loss`: The Forecasting Mechanism

This function tells the latent space: *"You must contain enough information to help the model predict the future text description."* This is technically a **State-Conditioned Generative Loss**.

| Action | Purpose | Detail |
| --- | --- | --- |
| **`proj_state`** | Latent  Soft Prompt | We transform the 16-dim state into `prefix_tokens` (high-dimensional vectors). These vectors act as a "hidden context" that precedes the text. |
| **`inputs_embeds`** | Bypassing the Vocabulary | Instead of passing token IDs, we pass raw embedding vectors to the LLM. This allows us to inject our "Soft Prompt" directly into the Transformer's first layer. |
| **Prefix Injection** | Guiding the LLM | We concatenate our projected state to the front of the actual text embeddings. |
| **Cross-Entropy Loss** | Next-Token Prediction | We calculate how well the LLM predicts the actual `text` given our "Soft Prompt." |

```python
    def text_loss(self, x_t: torch.Tensor, text: str) -> torch.Tensor:
        # PROJECT LATENT TO "SOFT PROMPT"
        # We take the 16-dim vector and expand it to [prefix_tokens, 1536].
        # This creates "pseudo-tokens" that represent our belief state.
        prefix_emb = self.proj_state(x_t).view(1, self.cfg.prefix_tokens, -1)
        
        # TOKENIZE AND EMBED THE TARGET TEXT
        prompt = f"Given this belief state, generate a textual forecast.\nDate: 2025-01-01\n"
        full_prompt = prompt + text
        target_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Convert IDs to their standard 1536-dim embeddings in the model.
        inputs_embeds = self.model.model.embed_tokens(target_ids)
        
        # ATTACH THE LATENT STATE TO THE INPUT
        # We glue our latent-prefix to the front of the text embeddings.Now the LLM "sees" the latent state before it reads the prompt.
        inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)
        
        # ALIGN LABELS
        # We need to tell the loss function to ignore the prefix tokens when calculating error. We use -100 (the standard ignore_index in PyTorch) for the prefix positions.
        labels = target_ids.clone()
        labels = torch.cat([torch.full((1, self.cfg.prefix_tokens), -100, device=self.device), labels], dim=1)

        # GENERATIVE LOSS
        # Standard Causal Language Modeling loss (Cross-Entropy).
        out = self.model(inputs_embeds=inputs_embeds, labels=labels)
        return out.loss

```

### Why this is critical for Goal-Conditioning

Because `text_loss` connects the latent state to the model's output, when we **steer** the state toward a Goal in `GoalLBS`, this loss ensures that the steered state actually results in steered text. It creates the semantic bridge.

---

## `LBS.kl_loss`: The Consistency Regulator

This method calculates the "distance" between two probability distributions: the **Prior** (what the GRU predicted) and the **Posterior** (what the Filter updated).

The `kl_loss` (Kullback-Leibler Divergence) is the most mathematically "dense" part of the network, but it serves a vital role as the **information regulator**.

In an RSSM, the model can "cheat" by ignoring the temporal dynamics (the GRU) and just using the current observation to create the state. `kl_loss` forces the model to ensure that its current belief is consistent with its previous expectations.

| Concept | Purpose | Analogy |
| --- | --- | --- |
| **KL Divergence** | Measures "surprise." | How much "new" information did we gain from the observation that the prior didn't expect? |
| **Gaussian Formula** | Closed-form math. | Since we assume our latents are Normal distributions, we can calculate the distance exactly without sampling. |
| **Free Nats** | Information Slack. | We don't want the Prior and Posterior to be *identical* (that would mean we aren't learning from data). "Free nats" allows the model a certain amount of "free information" before it starts getting penalized. |
| **Annealing** | Gradual Constraint. | We start with high "Free Nats" to let the model explore, then slowly tighten the constraint as training progresses. |

---

### Step-by-Step KL Logic

```python
    def kl_loss(self, prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps):
        # THE GAUSSIAN KL FORMULA
        # This equation calculates the distance between two Normal distributions.
        # It penalizes:
        #   - Means that are far apart (prior_mu vs post_mu)
        #   - Variances that differ (prior_logvar vs post_logvar)
        kl = -0.5 * torch.sum(1 + post_logvar - prior_logvar
                              - (post_mu - prior_mu).pow(2) / prior_logvar.exp()
                              - post_logvar.exp() / prior_logvar.exp())
        
        # CALCULATE "FREE NATS" (The Slack)
        # 'Nats' is a unit of information (like bits). 
        # We allow the model to have some divergence without penalty. This prevents "Posterior Collapse" (where the model ignores the Prior). Then we anneal this (reduce it) over time as the model gets smarter.
        free_nats = self.cfg.kl_free_nats * max(0.0, 1.0 - step / total_steps)
        
        # CLAMPING
        # We only penalize the model if the KL is GREATER than our allowed free nats.
        return torch.clamp(kl - free_nats, min=0.0)

```

---

### Why this is critical for your Goal-Conditioning

This is where our **Steering** actually exerts its force.
When we modify the `prior_mu` in our `GoalLBS` class to point toward a goal, the `kl_loss` tries to pull the `post_mu` (the actual state) toward that steered prior.

* **Without KL Loss:** The model would ignore our steering and just follow the data.
* **With KL Loss:** The model is "guilt-tripped" into making its state look like the goal state we've injected into the prior.

---

## `LBS.training_step`: The Orchestrator

This method ties the encoding, filtering, and loss calculations together into a single differentiable unit.

The `training_step` is the **orchestrator**. It doesn't contain new logic, but it defines the *execution flow*—how a single observation is processed, how the losses are aggregated, and how the state is passed forward to the next moment in time.

We can think of this as a single frame in a movie; this method defines what happens in that frame and prepares the "film" for the next one.

| Action | Purpose | Detail |
| --- | --- | --- |
| **Observation Encoding** | Sensor Input. | Calls `encode_text` to get . |
| **State Transition** | Moving the "Mind". | Calls `filter_step` to produce the new latent state . |
| **Loss Aggregation** | Multi-Objective Optimization. | Combines Value, Text, and KL losses using weight coefficients (). |
| **Detachment** | Memory Management. | Returns detached states () to prevent the computational graph from growing infinitely during long sequences (though we discussed keeping them attached for full BPTT earlier). |

---

### Step-by-Step Logic

```python
    def training_step(self, y_t, text, x_prev, h_prev, step, total_steps):
        # PERCEIVE THE ENVIRONMENT
        # Turn the current text description into a semantic vector.
        s_t = self.encode_text(text)
        
        # UPDATE THE INTERNAL BELIEF
        # Predict the next state (prior) and refine it with observations (posterior).
        # x_t: The sampled latent vector (used for decoding).
        # prior/post: The distribution parameters (used for KL loss).
        # h_t: The updated GRU memory.
        x_t, prior, post, h_t = self.filter_step(x_prev, h_prev, y_t, s_t)
        
        # EVALUATE THE BELIEF (The Three Losses)
        # L_val: Does x_t correctly represent the numerical data (y_t)?
        L_val = self.value_loss(x_t, y_t)
        
        # L_text: Can the model generate the text description given x_t?
        # Note: we detach x_t here to isolate the text loss gradients (a common trick in RSSMs to stabilize training).
        L_text = self.text_loss(x_t.detach(), text)
        
        # L_kl: Is our update consistent with our history/dynamics?
        L_kl = self.kl_loss(prior[0], prior[1], post[0], post[1], step, total_steps)
        
        # WEIGHTED TOTAL LOSS
        # We use alpha weights (e.g., alpha_text=10, alpha_val=1) to balance how much the model cares about numbers vs. text vs. consistency.
        loss = self.cfg.alpha_val * L_val + self.cfg.alpha_text * L_text + self.cfg.alpha_kl * L_kl
        
        # HAND-OFF
        # We return the loss for backprop, and the detached states to be used as the 'prev' inputs for the NEXT timestep.
        return loss, x_t.detach(), h_t.detach(), (L_val.item(), L_text.item(), L_kl.item())

```

---

### LBS Big Picture Checkpoint

We have finished with the base `LBS` class. We now have a model that can:

1. **Read** text into a 16-dim state.
2. **Remember** history via a GRU.
3. **Predict** numbers and text from that state.
4. **Balance** its internal imagination with external reality via KL divergence.

---

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

### Step-by-Step GoalLBS Forward Step Logic

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

We now have semi-functional novel architecture:

1. **`LBS`**: The world model that understands history, numbers, and text.
2. **`GoalLBS`**: The control layer that nudges the world model toward a specific target.
3. **Bayesian Filtering**: The math that ensures the "nudging" remains consistent with reality.
4. **Prefix Tuning**: The interface that allows the state space to "talk" to the Transformer.
