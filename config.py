import torch


class CFG:
    latent_dim = 16
    gru_layers = 1
    sum_tokens = 8                # K <SUM> tokens
    prefix_tokens = 8             # projected state tokens
    lora_r = 8
    lora_alpha = 16
    kl_free_nats = 2.5
    alpha_val = 1.0
    alpha_text = 0.1
    alpha_kl = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_type = torch.bfloat16