import torch
import torch.optim as optim
from goal import LBS, GoalLBS

def run_test():
    print("--- Initializing Goal-Conditioned LBS Test Bench ---")
    
    # Setup Model
    lbs = LBS("ibm-granite/granite-4.0-h-350m")
    goal_lbs = GoalLBS(lbs)
    optimizer = optim.AdamW(goal_lbs.parameters(), lr=1e-4)

    # Create Dummy Sequence Data
    # A sequence of 3 steps: (Value, Text)
    data_sequence = [
        (10.0, "The system state is currently stable at baseline."),
        (12.5, "A slight upward trend in energy output is observed."),
        (15.0, "Energy output has reached the primary efficiency threshold.")
    ]
    goal_text = "Achieve maximum energy efficiency and system stability."

    print(f"\nEncoding Goal: '{goal_text}'")
    x_goal = goal_lbs.encode_goal(goal_text)
    
    # Training Pass
    print("\nStarting Training Step...")
    optimizer.zero_grad()
    
    # Initialize hidden states [num_layers, batch, dim]
    batch_size = 1
    h_prev = torch.zeros(lbs.cfg.gru_layers, batch_size, lbs.cfg.latent_dim, 
                         device=lbs.device, dtype=lbs.cfg.d_type)
    x_prev = torch.zeros(batch_size, lbs.cfg.latent_dim, 
                         device=lbs.device, dtype=lbs.cfg.d_type)
    
    total_loss = 0
    seq_len = len(data_sequence)

    for i, (val, txt) in enumerate(data_sequence):
        # Format numerical data to [batch, 1]
        y_t = torch.tensor([[val]], device=lbs.device, dtype=lbs.cfg.d_type)
        
        # Forward pass (training=True keeps the graph attached)
        x_t, h_t, metrics, _ = goal_lbs.forward_step(
            y_t=y_t,
            text_t=txt,
            x_prev=x_prev,
            h_prev=h_prev,
            x_goal=x_goal,
            step=i,
            total_steps=seq_len,
            training=True
        )
        
        total_loss += metrics['loss']
        
        # Update states for next step (Keep attached!)
        x_prev = x_t
        h_prev = h_t
        
        print(f"  Step {i+1} | Loss: {metrics['loss'].item():.4f} | KL: {metrics['kl']:.4f}")

    # Backward Pass & Gradient Verification
    print("\nComputing Gradients...")
    total_loss.backward()

    # Verify that gradients reached the start of the chain
    goal_enc_grad = goal_lbs.goal_encoder[0].weight.grad
    gru_grad = lbs.gru.weight_hh_l0.grad
    
    if goal_enc_grad is not None and gru_grad is not None:
        print("✅ Success: Gradients successfully propagated to Goal Encoder and GRU.")
        print(f"   Goal Encoder Grad Norm: {goal_enc_grad.norm().item():.6f}")
        print(f"   GRU Grad Norm: {gru_grad.norm().item():.6f}")
    else:
        print("❌ Error: Gradients are None. Check for accidental .detach() or no_grad() blocks.")

    torch.nn.utils.clip_grad_norm_(goal_lbs.parameters(), max_norm=1.0)
    
    optimizer.step()

    # 5. Inference Test (The Plan)
    print("\nTesting Plan Generation (Latent Rollout)...")
    try:
        plan = goal_lbs.generate_plan(x_prev.squeeze(0), steps_ahead=2)
        print("\nGenerated Plan Sample:")
        print("-" * 30)
        print(plan)
        print("-" * 30)
    except Exception as e:
        print(f"❌ Planning Failed: {str(e)}")

if __name__ == "__main__":
    run_test()