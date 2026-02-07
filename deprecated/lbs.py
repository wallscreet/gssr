

#! Replaced this forward pass 
# def forward_step(self, y_t, text_t, x_prev, h_prev, x_goal, step, total_steps, training=True):
    #     """
    #     This method executes one temporal step of the model while applying the latent "nudge" toward the goal.
    #     """
    #     # SEMANTIC ENCODING
    #     # The model looks at the current text and compresses it to a 16-dim observation.
    #     s_t = self.lbs.encode_text(text_t)

    #     # STATE TRANSITION (The "Mind" moves forward)
    #     # We pass the previous sample through the GRU.
    #     # h_t: [num_layers, batch, latent_dim]
    #     _, h_t = self.lbs.gru(x_prev.unsqueeze(1), h_prev)
        
    #     # GENERATE RAW PRIOR
    #     # Extract the top-layer hidden state and guess the next distribution.
    #     prior_latent = h_t[-1] 
    #     prior = self.lbs.mlp_prior(prior_latent)
    #     prior_mu, prior_logvar = prior.chunk(2, dim=-1)

    #     # APPLY THE STEERING (Crucial Step)
    #     # We nudge the predicted mean toward x_goal.
    #     # This modification is "recorded" in the gradient graph.
    #     steered_mu = self.steer_prior(prior_mu, x_goal, step, total_steps)

    #     # GENERATE POSTERIOR (The "Fact-Check")
    #     # Combine: RNN Context (h_t) + Data (y_t) + Semantic Obs (s_t).
    #     post_in = torch.cat([prior_latent, y_t, s_t], dim=-1)
    #     post = self.lbs.mlp_post(post_in)
    #     post_mu, post_logvar = post.chunk(2, dim=-1)

    #     # REPARAMETERIZATION SAMPLE
    #     # Sample the actual state x_t from the posterior.
    #     std = torch.exp(0.5 * post_logvar)
    #     eps = torch.randn_like(std)
    #     x_t = post_mu + eps * std

    #     # LOSS CALCULATION
    #     # Note: L_kl uses 'steered_mu'! This forces the model to 
    #     # minimize the distance between its belief and the GOAL.
    #     L_val = self.lbs.value_loss(x_t, y_t.mean(dim=-1)).mean()
    #     L_text = self.lbs.text_loss(x_t, text_t).mean()
    #     L_kl = self.lbs.kl_loss(steered_mu, prior_logvar, post_mu, post_logvar, step, total_steps).mean()

    #     # Combine with weight coefficients
    #     loss = (self.lbs.cfg.alpha_val * L_val +
    #             self.lbs.cfg.alpha_text * L_text +
    #             self.lbs.cfg.alpha_kl * L_kl)

    #     # HAND-OFF
    #     # If training, we return the attached tensors so gradients flow back 
    #     # across time steps (BPTT).
    #     return (x_t if training else x_t.detach(), 
    #             h_t if training else h_t.detach(), 
    #             {'loss': loss, 'val': L_val.item(), 'text': L_text.item(), 'kl': L_kl.item()},
    #             steered_mu)