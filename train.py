import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
from goal import LBS, GoalLBS
from utils import debug_tensor

torch.autograd.set_detect_anomaly(True)

def train_lbs(model_path="ibm-granite/granite-4.0-h-350m", epochs=10, seq_len=50, lr=1e-4):
    print("Starting LBS training run...")
    
    lbs = LBS(model_path)
    goal_lbs = GoalLBS(lbs)
    
    optimizer = optim.AdamW(goal_lbs.parameters(), lr=lr)
    
    # Synthetic data generator (vector y_t [1, latent_dim] for richer obs)
    def generate_synthetic_sequence(length):
        data = []
        for t in range(length):
            y_scalar = math.sin(t * 0.5) + 0.1 * torch.randn(1).item()
            y_t = torch.tensor([[y_scalar]], device=lbs.device, dtype=lbs.cfg.d_type)  # [1, 1]
            text_t = f"At time {t}, value ≈ {y_scalar:.2f}. Predict next."
            data.append((y_t, text_t))
        return data
    
    # Goal text
    goal_text = "Achieve stable values close to 1.0 across all dimensions in 10 steps"
    x_goal = goal_lbs.encode_goal(goal_text).detach()
    print(f"Encoded goal shape: {x_goal.shape}")

    # Initial states
    x_prev = torch.zeros(1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
    #debug_tensor(x_prev, "train.train_lbs x_prev")
    h_prev = torch.zeros(lbs.cfg.gru_layers, 1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
    #debug_tensor(h_prev, "train.train_lbs h_prev")
    
    best_loss = float('inf')
    total_training_steps = epochs * seq_len
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        data_sequence = generate_synthetic_sequence(seq_len)
        
        epoch_loss = 0
        x_prev = torch.zeros(1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
        h_prev = torch.zeros(lbs.cfg.gru_layers, 1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
        
        for t, (y_t, text_t) in enumerate(data_sequence):
            optimizer.zero_grad()
            
            x_t, h_t, loss, breakdown, steered_mu = goal_lbs.forward_step(
                y_t=y_t,
                text_t=text_t,
                x_prev=x_prev,
                h_prev=h_prev,
                x_goal=x_goal,
                step=t,
                total_steps=seq_len,
                current_epoch_step=epoch * seq_len + t,
                training=True
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(goal_lbs.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Detach carry-over to break graph and save memory
            x_prev = x_t.detach()
            h_prev = h_t.detach()
            
            epoch_loss += loss.item()
            if t % 2 == 0:
                print(f"  Step {t}: Total Loss={loss.item():.4f}, Val={breakdown['val']:.4f}, Text={breakdown['text']:.4f}, KL={breakdown['kl']:.4f}")

        avg_loss = epoch_loss / seq_len
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'x_final': x_prev,
                'h_final': h_prev,
                'goal_encoder_state': goal_lbs.goal_encoder.state_dict(),
            }, 'lbs_state.pth')
            print("  Saved best state.")

    # Final plan demo with best state
    print("\nGenerating plan from final state...")
    plan = goal_lbs.generate_plan(x_prev.squeeze(0), steps_ahead=5)
    print(plan)

if __name__ == "__main__":
    train_lbs(epochs=1, seq_len=8, lr=1e-4)