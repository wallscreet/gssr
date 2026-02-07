import torch

def train_goal_conditioned_sequence(model, optimizer, sequence_data, goal_text):
    model.train()
    optimizer.zero_grad()
    
    # 1. ENCODE THE GOAL (Once per sequence)
    x_goal = model.encode_goal(goal_text)
    
    # 2. INITIALIZE STATES
    batch_size = 1
    h_prev = torch.zeros(model.lbs.cfg.gru_layers, batch_size, model.latent_dim, 
                         device=model.device, dtype=model.lbs.cfg.d_type)
    x_prev = torch.zeros(batch_size, model.latent_dim, 
                         device=model.device, dtype=model.lbs.cfg.d_type)
    
    accumulated_loss = 0
    seq_len = len(sequence_data)
    
    # Track steps globally for the Annealing Ratio
    if not hasattr(model, 'global_step'): model.global_step = 0

    # 3. TEMPORAL ROLLOUT
    for i, (val, text) in enumerate(sequence_data):
        y_t = torch.tensor([[val]], device=model.device, dtype=model.lbs.cfg.d_type)
        
        # We pass global_step to the forward_step for the KL Annealing
        x_t, h_t, metrics, _ = model.forward_step(
            y_t=y_t,
            text_t=text,
            x_prev=x_prev,
            h_prev=h_prev,
            x_goal=x_goal,
            step=i,
            total_steps=seq_len,
            current_epoch_step=model.global_step,
            training=True
        )
        
        accumulated_loss += metrics['loss']
        
        # Update states for the next temporal step
        # Note: We keep them attached to the graph for BPTT
        x_prev = x_t
        h_prev = h_t
        model.global_step += 1

    # 4. BACKPROPAGATE THROUGH TIME (BPTT)
    # This flows gradients through the entire sequence length
    accumulated_loss.backward()
    
    # 5. THE STABILITY ANCHOR
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return accumulated_loss.item() / seq_len