import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import debug_module, debug_tensor
from config import CFG


# LBS core model
class LBS(nn.Module):
    def __init__(self, model_path="ibm-granite/granite-4.0-h-1b"):  # Use small model for test
        super().__init__()
        self.cfg = CFG()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.cfg.device,
            torch_dtype=self.cfg.d_type,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = self.cfg.device

        # SSM (GRU)
        self.gru = nn.GRU(self.cfg.latent_dim, self.cfg.latent_dim,
                        num_layers=self.cfg.gru_layers, batch_first=True)
        self.gru = self.gru.to(self.device, self.cfg.d_type)
        debug_module(self.gru, "gru")
        
        # torch.Size([32, 16])
        self.mlp_prior = nn.Linear(self.cfg.latent_dim, self.cfg.latent_dim * 2)
        self.mlp_prior = self.mlp_prior.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_prior, "mlp_prior")
        
        # torch.Size([32, 33])
        self.mlp_post = nn.Linear(self.cfg.latent_dim * 2 + 1, self.cfg.latent_dim * 2)
        self.mlp_post = self.mlp_post.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_post, "mlp_post")
        
        # torch.Size([1, 16])
        self.mlp_val = nn.Linear(self.cfg.latent_dim, 1)
        self.mlp_val = self.mlp_val.to(self.device, self.cfg.d_type)
        debug_module(self.mlp_val, "mlp_val")
        
        # torch.Size([16, 1536])
        self.proj_sum = nn.Linear(self.model.config.hidden_size, self.cfg.latent_dim)
        self.proj_sum = self.proj_sum.to(self.device, self.cfg.d_type)
        debug_module(self.proj_sum, "proj_sum")
        
        # torch.Size([12288, 16])
        self.proj_state = nn.Linear(self.cfg.latent_dim, self.cfg.prefix_tokens * self.model.config.hidden_size)
        self.proj_state = self.proj_state.to(self.device, self.cfg.d_type)
        debug_module(self.proj_state, "proj_state")

        # token ids (long is fine, no dtype issue)
        self.sum_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(f"<SUM{i}>") for i in range(self.cfg.sum_tokens)],
            dtype=torch.long, device=self.device
        )
        
    def encode_text(self, text: str) -> torch.Tensor:
        prompt = f"Information encoded into a sequence of vectors.\n{text}\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids = torch.cat([input_ids, self.sum_ids.unsqueeze(0)], dim=1)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        sum_hidden = hidden[0, -self.cfg.sum_tokens:]
        s = self.proj_sum(sum_hidden)
        s = s.mean(dim=0, keepdim=True)
        return s

    def filter_step(self, x_prev, h_prev, y_t, s_t):
        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)

        gru_in = x_prev.unsqueeze(1)
        _, h_t = self.gru(gru_in, h_prev)
        prior = self.mlp_prior(h_t.squeeze(1))
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        post_in = torch.cat([h_t.squeeze(1), y_t.unsqueeze(1), s_t], dim=-1)  # fix y_t to [1,1]
        post = self.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)

        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std, dtype=self.cfg.d_type)
        x_t = post_mu + eps * std

        return x_t, (prior_mu, prior_logvar), (post_mu, post_logvar), h_t

    def value_loss(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        y_pred = self.mlp_val(x_t)
        return F.mse_loss(y_pred.squeeze(-1), y_t)

    def text_loss(self, x_t: torch.Tensor, text: str) -> torch.Tensor:
        prefix_emb = self.proj_state(x_t).view(1, self.cfg.prefix_tokens, -1)
        prompt = f"Given this belief state, generate a textual forecast.\nDate: 2025-01-01\n"
        full_prompt = prompt + text
        target_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)

        inputs_embeds = self.model.model.embed_tokens(target_ids)
        debug_tensor(inputs_embeds, "inputs_embeds")
        inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)
        debug_tensor(inputs_embeds, "inputs_embeds after concat")
        labels = target_ids.clone()
        labels = torch.cat([torch.full((1, self.cfg.prefix_tokens), -100, device=self.device), labels], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, labels=labels)
        return out.loss

    def kl_loss(self, prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps):
        kl = -0.5 * torch.sum(1 + post_logvar - prior_logvar
                              - (post_mu - prior_mu).pow(2) / prior_logvar.exp()
                              - post_logvar.exp() / prior_logvar.exp())
        free_nats = self.cfg.kl_free_nats * max(0.0, 1.0 - step / total_steps)
        return torch.clamp(kl - free_nats, min=0.0)

    def training_step(self, y_t, text, x_prev, h_prev, step, total_steps):
        s_t = self.encode_text(text)
        x_t, prior, post, h_t = self.filter_step(x_prev, h_prev, y_t, s_t)
        L_val = self.value_loss(x_t, y_t)
        L_text = self.text_loss(x_t.detach(), text)
        L_kl = self.kl_loss(prior[0], prior[1], post[0], post[1], step, total_steps)
        loss = self.cfg.alpha_val * L_val + self.cfg.alpha_text * L_text + self.cfg.alpha_kl * L_kl
        return loss, x_t.detach(), h_t.detach(), (L_val.item(), L_text.item(), L_kl.item())


# GoalLBS: Goal-Conditioned LBS
class GoalLBS(nn.Module):
    def __init__(self, base_lbs: LBS, goal_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.lbs = base_lbs
        self.device = base_lbs.device
        self.latent_dim = base_lbs.gru.hidden_size

        # Goal encoder: text -> target latent state
        if goal_encoder is None:
            self.goal_encoder = nn.Sequential(
                nn.Linear(base_lbs.model.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
                nn.Tanh()
            )
        else:
            self.goal_encoder = goal_encoder
        
        self.goal_encoder = self.goal_encoder.to(device=self.device, dtype=self.lbs.cfg.d_type)

        # Steering strength - how hard to pull toward goal (tunable)
        self.gamma = 0.5


    # Encode goal text -> target latent state x_goal
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        prompt = f"Goal: {goal_text}\nSummarize the desired future state."
        input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            hidden = self.lbs.model.model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        goal_emb = hidden[0, -1, :]  # last token
        x_goal = self.goal_encoder(goal_emb)
        return x_goal  # (latent_dim,)


    # Steering: pull toward x_goal with gating, annealing and clamping
    def steer_prior(self, prior_mu, x_goal, step=0, total_steps=5):
        gamma = self.gamma * (1 - step / total_steps)  # decay over steps
        error = x_goal - prior_mu
        gate = torch.sigmoid(prior_mu.abs().mean(dim=-1, keepdim=True))
        correction = gamma * gate * error
        steered_mu = prior_mu + correction
        
        norm = steered_mu.norm(dim=-1, keepdim=True)
        scale = torch.clamp(5.0 / (norm + 1e-8), max=1.0)
        steered_mu = steered_mu * scale
        
        return steered_mu


    # Forward
    def forward_step(self,
                 y_t: torch.Tensor,
                 text_t: str,
                 x_prev: torch.Tensor,
                 h_prev: torch.Tensor,
                 x_goal: torch.Tensor,
                 step: int,
                 total_steps: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns: x_t, h_t, losses
        """
        # Force y_t to [batch, latent_dim] for concat
        # if y_t.dim() == 0:
        #     y_t = y_t.unsqueeze(0).unsqueeze(-1)  # scalar → [1,1]
        # elif y_t.dim() == 1:
        #     y_t = y_t.unsqueeze(0)  # [dim] → [1, dim]
        # elif y_t.shape[-1] == 1:
        #     y_t = y_t.repeat(1, self.lbs.cfg.latent_dim)  # [1,1] → [1, latent_dim] by repeat

        s_t = self.lbs.encode_text(text_t)

        # Prior from SSM
        _, h_t = self.lbs.gru(x_prev.unsqueeze(1), h_prev)  # unsqueeze to add seq dim=1
        debug_tensor(h_t, "h_t after gru")
        debug_tensor(h_t.squeeze(0), "h_t squeezed")
        
        prior = self.lbs.mlp_prior(h_t.squeeze(1))
        debug_tensor(prior, "prior")
        
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        # STEER PRIOR toward goal
        prior_mu = self.steer_prior(prior_mu, x_goal)
        
        # debug print statements
        debug_tensor(y_t, "y_t")
        debug_tensor(s_t, "s_t")
        print("post_in expected concat dim:", h_t.squeeze(0).shape[-1] + y_t.unsqueeze(0).shape[-1] + s_t.shape[-1])

        # Posterior (uses steered prior in KL)
        post_in = torch.cat([h_t.squeeze(1), y_t, s_t], dim=-1)  # all now [batch, latent_dim]
        debug_tensor(post_in, "post_in")
        post = self.lbs.mlp_post(post_in)
        debug_tensor(post, "post")
        post_mu, post_logvar = post.chunk(2, dim=-1)

        # Sample
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        x_t = post_mu + eps * std

        # Losses
        L_val = self.lbs.value_loss(x_t, y_t.mean(dim=-1))  # average y_t if [1, latent_dim]
        L_text = self.lbs.text_loss(x_t.detach(), text_t)
        L_kl = self.lbs.kl_loss(prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps)

        loss = (self.lbs.cfg.alpha_val * L_val +
                self.lbs.cfg.alpha_text * L_text +
                self.lbs.cfg.alpha_kl * L_kl)

        return x_t.detach(), h_t.detach(), {
            'loss': loss, 'val': L_val.item(), 'text': L_text.item(), 'kl': L_kl.item()
        }, prior_mu


    # Generate Plan from Final State
    def generate_plan(self, x_final: torch.Tensor, steps_ahead: int = 5) -> str:
        plan = "Plan to achieve goal:\n"
        x = x_final.unsqueeze(0)
        h = torch.zeros(self.lbs.gru.num_layers, 1, self.latent_dim, device=self.device)

        for step in range(steps_ahead):
            # Project state -> prefix
            prefix_emb = self.lbs.proj_state(x).view(1, self.lbs.cfg.prefix_tokens, -1)
            prompt = f"Given this belief state, generate the next step in the plan.\nDate: Step {step+1}\n"
            input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            inputs_embeds = self.lbs.model.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)

            output = self.lbs.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.lbs.tokenizer.eos_token_id
            )
            step_text = self.lbs.tokenizer.decode(output[0].split(input_ids.shape[1] + self.lbs.cfg.prefix_tokens)[-1], skip_special_tokens=True)
            plan += f"Step {step+1}: {step_text}\n"

            # Roll forward (no obs)
            with torch.no_grad():
                _, h = self.lbs.gru(x.unsqueeze(0), h)
                prior = self.lbs.mlp_prior(h.squeeze(0))
                prior_mu, _ = prior.chunk(2, dim=-1)
                x = prior_mu  # deterministic roll

        return plan


def main():
    lbs = LBS()
    #goal = GoalLBS()

if __name__ == "__main__":
    main()