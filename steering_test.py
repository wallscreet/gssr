import torch
from torch.optim import AdamW
import math
from goal import LBS, GoalLBS
from utils import debug_tensor


# Test script
def test_lbs():
    lbs = LBS(model_path="ibm-granite/granite-4.0-h-350m")
    goal_lbs = GoalLBS(lbs)

    num_steps = 10
    total_steps = 100
    optimizer = AdamW(goal_lbs.parameters(), lr=1e-4)
    
    x_prev = torch.zeros(1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
    h_prev = torch.zeros(lbs.cfg.gru_layers, 1, lbs.cfg.latent_dim, device=lbs.device, dtype=lbs.cfg.d_type)
    
    goal_text = "Reach a value close to 1.0 in 5 steps"
    x_goal = goal_lbs.encode_goal(goal_text)

    losses = []
    for t in range(num_steps):
        y_scalar = math.sin(t * 0.5) + 0.1 * torch.randn(1).item()
        y_t = torch.tensor([y_scalar], device=lbs.device, dtype=lbs.cfg.d_type).unsqueeze(0)  # [1,1] — no extra unsqueeze(-1)
        debug_tensor(y_t, "y_t in test lbs")
        
        text_t = f"At time {t}, the value is approximately {y_scalar:.2f}. Predict next."

        loss, x_t, h_t, breakdown = goal_lbs.forward_step(
            y_t, text_t, x_prev, h_prev, x_goal, t, total_steps
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        x_prev = x_t
        h_prev = h_t
        
        losses.append(breakdown)
        print(f"Step {t}: Loss={loss.item():.4f}, Val={breakdown['val']:.4f}, Text={breakdown['text']:.4f}, KL={breakdown['kl']:.4f}")

    plan = goal_lbs.generate_plan(x_t)
    print("\nGenerated Plan:\n", plan)

if __name__ == "__main__":
    test_lbs()