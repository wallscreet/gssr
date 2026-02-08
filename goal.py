import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import debug_module, debug_tensor, init_weights
from config import CFG


# LBS core model
class LBS(nn.Module):
    def __init__(self, model_path="ibm-granite/granite-4.0-h-350m"):  # Use small model for test
        super().__init__()
        self.cfg = CFG()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.cfg.device,
            torch_dtype=self.cfg.d_type,
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.device = self.cfg.device

        # SSM (GRU)
        # This keeps track of history. It receives the previous latent sample (x_prev) 
        # and moves the hidden state (h_prev) forward.
        self.gru = nn.GRU(self.cfg.latent_dim, self.cfg.latent_dim,
                        num_layers=self.cfg.gru_layers, batch_first=True)
        self.gru = self.gru.to(self.device, self.cfg.d_type)
        debug_module(self.gru, "gru")
        
        # mlp_prior is the TRANSITION PRIOR (Distribution over the "future")
        # Takes the GRU hidden state and outputs a mu and logvar.
        # Shape: [latent_dim] -> [latent_dim * 2] (for mu and logvar)
        # torch.Size([32, 16])
        self.mlp_prior = nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim * 2)
        #! Init weights from utils
        self.mlp_prior.apply(init_weights)
        self.mlp_prior = self.mlp_prior.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_prior, "mlp_prior")
        
        # mlp_post is the POSTERIOR FILTER (Distribution over "now")
        # Combines: 1. RNN State (History) + 2. Numerical Obs (y_t) + 3. Semantic Obs (s_t)
        # It corrects the prior based on what we just observed.
        # torch.Size([32, 33])
        self.mlp_post = nn.Linear(self.cfg.latent_dim * 2 + 1, self.cfg.latent_dim * 2)
        #! New init weights from utils
        self.mlp_post.apply(init_weights)
        self.mlp_post = self.mlp_post.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_post, "mlp_post")
        
        # mlp_val is the REWARD/VALUE ESTIMATOR
        # Decodes the latent state into a single float. Used for MSE loss against data.
        # torch.Size([1, 16])
        self.mlp_val = nn.Linear(self.cfg.latent_dim, 1)
        self.mlp_val = self.mlp_val.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_val, "mlp_val")
        
        # proj_sum is the SEMANTIC COMPRESSION HEAD
        # Maps LLM hidden_size (1536) -> latent_dim (16). 
        # This is the "eye" of the model, seeing the LLM's interpretation of text.
        # torch.Size([16, 1536])
        self.proj_sum = nn.Linear(self.model.config.hidden_size, self.cfg.latent_dim)
        self.proj_sum = self.proj_sum.to(self.device, self.cfg.d_type)
        debug_module(self.proj_sum, "proj_sum")
        
        # proj_state is the SEMANTIC GENERATION HEAD
        # Maps latent_dim (16) -> (Prefix_Tokens * hidden_size).
        # This allows the latent state to "speak" back to the LLM.
        # torch.Size([12288, 16])
        self.proj_state = nn.Linear(self.cfg.latent_dim, self.cfg.prefix_tokens * self.model.config.hidden_size)
        self.proj_state = self.proj_state.to(self.device, self.cfg.d_type)
        debug_module(self.proj_state, "proj_state")

        # sum_ids are the INFORMATION BOTTLE-NECK TOKENS
        # Pre-calculated IDs for <SUM0>, <SUM1>... to extract concentrated embeddings.
        # token ids (no dtype conflicts, long is fine)
        self.sum_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(f"<SUM{i}>") for i in range(self.cfg.sum_tokens)],
            dtype=torch.long, device=self.device
        )
            
    def encode_text(self, text: str) -> torch.Tensor:
        """This method transforms a high-level text description into a compact semantic vector () that the Bayesian filter can digest. In this step, we are bridging the gap between raw text and our latent space. This is the **Observation Model** (or "Encoder") of our system.
        """
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
        sum_hidden = hidden[0, -self.cfg.sum_tokens:]
        # PROJECT TO LATENT SPACE
        # Map 1536 -> 16.
        s = self.proj_sum(sum_hidden)
        # MEAN POOLING
        # Collapse the summary tokens into one 16-dimensional semantic observation.
        s = s.mean(dim=0, keepdim=True)
        return s

    def filter_step(self, x_prev, h_prev, y_t, s_t):
        """
        This method performs two critical roles: it generates a **Prior** (guessing the next state based on history) and a **Posterior** (refining that guess with actual evidence).
        """
        # SHAPE STABILIZATION
        # Ensuring inputs are tensors with batch dimensions [1, dim] to prevent broadcasting errors during concatenation.
        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)
            #debug_tensor(y_t, "filter y_t unsqueeze")
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        # DETERMINISTIC TRANSITION (RNN)
        # We pass the previous latent sample (x_prev) into the GRU. h_t now contains the "temporal context" updated with the latest step.
        gru_in = x_prev.unsqueeze(1)
        _, h_t = self.gru(gru_in, h_prev)
        #debug_tensor(h_t, "Temporal Context h_t")
        # GENERATE THE PRIOR (The "Guess")
        # We use the GRU's current memory to predict the distribution of the next state.
        # We split (or chunk) the output into Mean (mu) and Log-Variance (logvar).
        prior = self.mlp_prior(h_t.squeeze(1))

        prior_mu, prior_logvar = prior.chunk(2, dim=-1)
        #! raise prior_logvar clamp min to -5, keep max @ 2
        prior_logvar = torch.clamp(prior_logvar, min=-5.0, max=2.0)
        # GENERATE THE POSTERIOR (The "Fact-Check")
        # We concatenate history (h_t), numerical data (y_t), and semantic data (s_t).
        post_in = torch.cat([h_t.squeeze(1), y_t.unsqueeze(1), s_t], dim=-1)
        post = self.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)
        #! raise post_logvar clamp min to -5
        post_logvar = torch.clamp(post_logvar, min=-5.0, max=2.0)
        # THE REPARAMETERIZATION TRICK
        # We sample a specific latent vector (x_t) from the posterior distribution.
        # Using 'eps' (random noise) ensures we can still backpropagate through the mean/std.
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std, dtype=self.cfg.d_type)
        x_t = post_mu + eps * std

        return x_t, (prior_mu, prior_logvar), (post_mu, post_logvar), h_t

    def value_loss(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """
        This function forces the 16-dimensional vector to contain enough information to reconstruct the numerical data point ().
        """
        y_pred = self.mlp_val(x_t)
        return F.mse_loss(y_pred.squeeze(-1), y_t)

    def text_loss(self, x_t: torch.Tensor, text: str) -> torch.Tensor:
        """
        This function tells the latent space: *"You must contain enough information to help the model predict the future text description and is technically a State-Conditioned Generative Loss.
        """
        # PROJECT LATENT TO "SOFT PROMPT"
        # We take the 16-dim vector and expand it to [prefix_tokens, 1536].
        # This creates "pseudo-tokens" that represent our belief state.
        #! exploding loss and kl div problem - lowering prefix_emb by an order of magnitude to observe consequence
        prefix_emb = self.proj_state(x_t).view(1, self.cfg.prefix_tokens, -1) * 0.1
        #debug_tensor(prefix_emb, "LBS.text_loss prefix_emb")
        # TOKENIZE AND EMBED THE TARGET TEXT
        prompt = f"Given this belief state, generate a textual forecast.\nDate: 2025-01-01\n"
        full_prompt = prompt + text
        target_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        # Convert IDs to their standard 1536-dim embeddings in the model.
        inputs_embeds = self.model.model.embed_tokens(target_ids)
        #debug_tensor(inputs_embeds, "LBS.text_loss inputs_embeds")
        # ATTACH THE LATENT STATE TO THE INPUT
        # We glue our latent-prefix to the front of the text embeddings.Now the LLM "sees" the latent state before it reads the prompt.
        inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)
        #debug_tensor(inputs_embeds, "LBS.text_loss inputs_embeds after concat")
        # ALIGN LABELS
        # We need to tell the loss function to ignore the prefix tokens when calculating error. We use -100 (the standard ignore_index in PyTorch) for the prefix positions.
        labels = target_ids.clone()
        labels = torch.cat([torch.full((1, self.cfg.prefix_tokens), -100, device=self.device), labels], dim=1)
        # GENERATIVE LOSS
        # Standard Causal Language Modeling loss (Cross-Entropy).
        out = self.model(inputs_embeds=inputs_embeds, labels=labels)
        return out.loss

    def kl_loss(self, prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps):
        """
        This method calculates the "distance" between two probability distributions: the **Prior** (what the GRU predicted) and the **Posterior** (what the Filter updated).
        """
        # THE GAUSSIAN KL FORMULA
        # This equation calculates the distance between two Normal distributions.
        # It penalizes:
        #   - Means that are far apart (prior_mu vs post_mu)
        #   - Variances that differ (prior_logvar vs post_logvar)
        kl = -0.5 * torch.sum(1 + post_logvar - prior_logvar
                              - (post_mu - prior_mu).pow(2) / prior_logvar.exp()
                              - post_logvar.exp() / prior_logvar.exp())
        # CALCULATE "FREE NATS"
        # We allow the model to have some divergence without penalty. This prevents "Posterior Collapse" (where the model ignores the Prior). Then we anneal this (reduce it) over time as the model gets smarter.
        free_nats = self.cfg.kl_free_nats * max(0.0, 1.0 - step / total_steps)
        # CLAMPING
        # We only penalize the model if the KL is GREATER than our allowed free nats.
        #return torch.clamp(kl - free_nats, min=0.0)
        #! Try not penalizing the KL until its above a certain threshold 
        #! This + initializing the dist weights with xavier normal caused exploding loss and kl. Try init dist zero plus this penalty
        return torch.max(kl, torch.tensor(0.5, device=self.device))

    def training_step(self, y_t, text, x_prev, h_prev, step, total_steps):
        """
        This method ties the encoding, filtering, and loss calculations together into a single differentiable unit.
        """
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


# GoalLBS: Goal-Conditioned LBS
class GoalLBS(nn.Module):
    """
    The GoalLBS is a wrapper. It takes our existing "World Model" (base_lbs) and adds a goal encoder that only looks at the target we want to achieve.
    """
    def __init__(self, base_lbs: LBS, goal_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.lbs = base_lbs
        self.device = base_lbs.device
        self.latent_dim = base_lbs.gru.hidden_size

        # THE GOAL ENCODER
        # This is a small MLP that bridges the model's world knowledge to our state space.
        # It takes a 1536-dim embedding and squashes it to 16-dim.
        if goal_encoder is None:
            self.goal_encoder = nn.Sequential(
                nn.Linear(base_lbs.model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
                nn.Tanh()
            )
        else:
            self.goal_encoder = goal_encoder
        
        # Ensure we match the precision (bfloat16) and device (cuda)
        self.goal_encoder = self.goal_encoder.to(device=self.device, dtype=self.lbs.cfg.d_type)
        debug_module(self.goal_encoder, "Goal Encoder")

        # Steering strength - how hard to pull toward goal (tunable)
        # High gamma = "The goal is everything, ignore reality."
        # Low gamma = "Try to achieve the goal, but stay grounded in data."
        self.gamma = 0.5

    # Encode goal text -> target latent state x_goal
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        """
        This method is only called **once** per generation or training sequence. It sets the "North Star" for the trajectory.
        """
        # PROMPT ENGINEERING
        # We prompt the model to summarize the 'future state' of the goal.
        prompt = f"Goal: {goal_text}\nSummarize the desired future state."
        input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        # LLM PASS (FROZEN)
        hidden = self.lbs.model.model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        #debug_tensor(hidden, "Goal.encode_goal hidden")
        # GLOBAL SUMMARIZATION
        # Take the last token's representation.
        goal_emb = hidden[0, -1, :]  # last token
        # LATENT MAPPING
        # Transform the embedding into our 16-dim control space.
        x_goal = self.goal_encoder(goal_emb)
        return x_goal  # (latent_dim,)

    # Steering: pull toward x_goal with gating, annealing and clamping
    def steer_prior(self, prior_mu, x_goal, step=0, total_steps=5):
        """
        This method calculates the "correction" needed to move the current trajectory toward the goal state.
        """
        # CALCULATE ANNEALED GAIN
        # We reduce the pull strength (gamma) as we approach the end of the sequence.
        # This helps the model stabilize its final state.
        gamma = self.gamma * (1 - step / total_steps)  # decay over steps
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
    
    #! -- NEW forward step from gemini recommendation
    def forward_step(self, y_t, text_t, x_prev, h_prev, x_goal, step, total_steps, current_epoch_step=0, training=True):
        """
        Executes one temporal step with latent "nudge" toward the goal and numerical stabilization.
        """
        # SEMANTIC ENCODING
        s_t = self.lbs.encode_text(text_t)
        #debug_tensor(s_t, "Goal.foward_step Fwd s_t")

        # STATE TRANSITION
        _, h_t = self.lbs.gru(x_prev.unsqueeze(1), h_prev)
        #debug_tensor(h_t, "Goal.forward_step h_t")
        
        # GENERATE RAW PRIOR
        prior_latent = h_t[-1]
        #debug_tensor(prior_latent, "Goal.forward_step prior_latent")
        prior = self.lbs.mlp_prior(prior_latent)
        #debug_tensor(prior, "Goal.forward_step prior")
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)
        
        # STABILITY: Clamp log-variance to prevent division-by-zero in KL
        prior_logvar = torch.clamp(prior_logvar, min=-5.0, max=2.0)

        # APPLY THE STEERING
        steered_mu = self.steer_prior(prior_mu, x_goal, step, total_steps)

        # GENERATE POSTERIOR
        post_in = torch.cat([prior_latent, y_t, s_t], dim=-1)
        #debug_tensor(post_in, "Goal.forward_step post_in")
        post = self.lbs.mlp_post(post_in)
        #debug_tensor(post, "Goal.forward_step post")
        post_mu, post_logvar = post.chunk(2, dim=-1)
        
        # STABILITY: Clamp log-variance
        post_logvar = torch.clamp(post_logvar, min=-5.0, max=2.0)

        # REPARAMETERIZATION SAMPLE
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        x_t = post_mu + eps * std

        # ANALYTICAL KL CALCULATION (More stable than sampling)
        prior_var = torch.exp(prior_logvar)
        post_var = torch.exp(post_logvar)
        
        # Distance between Steered Prior and actual Posterior
        kl_div = 0.5 * torch.sum(
            prior_logvar - post_logvar - 1.0 + 
            (post_var + (post_mu - steered_mu)**2) / prior_var, 
            dim=-1
        )

        # LINEAR KL ANNEALING
        # Ramp up alpha_kl over the first 500 steps to prevent 'KL Vanishing' or 'Explosion'
        warmup_steps = 500 
        anneal_ratio = min(1.0, current_epoch_step / warmup_steps)
        effective_alpha_kl = self.lbs.cfg.alpha_kl * anneal_ratio

        # MULTI-OBJECTIVE LOSS
        L_val = self.lbs.value_loss(x_t, y_t.mean(dim=-1)).mean()
        
        # STABILITY: Scale x_t down so it doesn't overwhelm the LLM's attention mechanism
        L_text = self.lbs.text_loss(x_t * 0.1, text_t).mean()
        
        # Use the 'Hinge' loss to keep KL from hitting zero exactly
        L_kl = torch.clamp(kl_div - self.lbs.cfg.kl_free_nats, min=0.5).mean()

        loss = (self.lbs.cfg.alpha_val * L_val +
                self.lbs.cfg.alpha_text * L_text +
                effective_alpha_kl * L_kl)

        return (
            x_t if training else x_t.detach(),
            h_t if training else h_t.detach(),
            loss,  # explicit scalar loss first
            {
                'val': L_val.item(),
                'text': L_text.item(),
                'kl': L_kl.item(),
                'anneal': anneal_ratio
            },
            steered_mu
        )

    def generate_plan(self, x_final: torch.Tensor, steps_ahead: int = 5) -> str:
        plan = "Plan to achieve goal:\n"
        x = x_final.unsqueeze(0).to(dtype=self.lbs.cfg.d_type)
        
        # Initialize an empty GRU hidden state for the rollout
        h = torch.zeros(self.lbs.gru.num_layers, 1, self.latent_dim, 
                        device=self.device, dtype=self.lbs.cfg.d_type)

        for step in range(steps_ahead):
            # TRANSLATE STATE TO SEMANTICS
            # Project our 16-dim belief into 'prefix_tokens' (e.g., 8 tokens * 1536 dim).
            prefix_emb = self.lbs.proj_state(x).view(1, self.lbs.cfg.prefix_tokens, -1)
            # PROMPT PREPARATION
            prompt = f"Given this belief state, generate the next step in the plan.\nStep {step+1}: "
            
            inputs = self.lbs.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            
            # HYBRID INPUT EMBEDDING
            # We combine our latent "Soft Prompt" with the textual "Hard Prompt".
            inputs_embeds = self.lbs.model.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)
            
            # Create an attention mask for the prefix + prompt
            # (Prefix tokens are always 1/active)
            prefix_mask = torch.ones((1, self.lbs.cfg.prefix_tokens), device=self.device)
            attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)
            # TEXT GENERATION
            # Granite generates text conditioned on the latent prefix.
            output = self.lbs.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.lbs.tokenizer.eos_token_id
            )
            
            # Decode only the NEW tokens (skipping the prompt and prefix)
            step_text = self.lbs.tokenizer.decode(output[0], skip_special_tokens=True)
            plan += f"Step {step+1}: {step_text}\n"

            # TEMPORAL ROLLOUT (The "Dream")
            # We move the state forward in time using only the GRU and the Prior MLP.
            # This is a 'closed-loop' simulation where the model predicts its own future.
            with torch.no_grad():
                _, h = self.lbs.gru(x.unsqueeze(0), h)
                prior = self.lbs.mlp_prior(h[-1]) # Use top layer
                prior_mu, _ = prior.chunk(2, dim=-1)
                x = prior_mu 

        return plan


def main():
    lbs = LBS()
    #goal = GoalLBS()

if __name__ == "__main__":
    main()