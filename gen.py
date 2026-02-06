import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GraniteSteerer:
    def __init__(self, model_path="ibm-granite/granite-4.0-h-1b", device="cuda"):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.model.eval()
        self.device = device
        self.d_state = 128

    def _prepare_inputs(self, prompt):
        chat = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(chat_text, return_tensors="pt").to(self.device)

    def generate_unsteered(self, prompt, max_new_tokens=50):
        """Standard generation without perturbation."""
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    def generate_steered(self, prompt, max_new_tokens=50, perturb_strength=0.05, target_layers=range(36, 40)):
        """Steered generation with perturbation on selected SSM layers."""
        inputs = self._prepare_inputs(prompt)
        
        with torch.no_grad():
            init_out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=True,
                return_dict_in_generate=True,
            )
        current_cache = init_out.past_key_values
        generated_ids = inputs.input_ids.clone()

        # Dummy goal vector (replace with real goal encoder later)
        goal = torch.randn(1, self.d_state, device=self.device) * perturb_strength

        for step in range(max_new_tokens):
            with torch.no_grad():
                step_out = self.model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=current_cache,
                    use_cache=True,
                    return_dict=True,
                )
            
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            current_cache = step_out.past_key_values
            
            # Apply perturbation
            if hasattr(current_cache, 'ssm_states'):
                ssm_list = current_cache.ssm_states
                for layer_idx in target_layers:
                    if layer_idx >= len(ssm_list):
                        continue
                    target_ssm = ssm_list[layer_idx]
                    
                    # flattening
                    if target_ssm.dim() >= 3:
                        current_flat = target_ssm.mean(dim=list(range(1, target_ssm.dim()-1)))  # → [1, 128]
                    else:
                        current_flat = target_ssm[:, -self.d_state:] if target_ssm.shape[-1] > self.d_state else target_ssm
                    
                    # Perturber logic
                    perturb_vec = torch.tanh(goal - current_flat)
                    perturb_strength_val = torch.sigmoid(current_flat.abs().mean(dim=-1))
                    perturbed_flat = current_flat + perturb_strength_val.unsqueeze(-1) * perturb_vec * perturb_strength
                    
                    # Broadcast back to original shape
                    target_shape = target_ssm.shape
                    perturbation = perturbed_flat.view(1, *(1,) * (len(target_shape)-2), self.d_state).expand(target_shape)
                    target_ssm.add_(perturbation)

        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return final_text

    def find_high_leverage_layers(self, prompt, test_strength=0.1):
        self.model.eval()
        inputs = self._prepare_inputs(prompt)
        
        # Force the model to generate 1 token to initialize the hybrid cache properly
        with torch.no_grad():
            init_out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
        # Extract the populated HybridMambaAttentionDynamicCache
        cache = init_out.past_key_values
        
        # Safety check: Ensure the cache isn't None before proceeding
        if cache is None:
            raise ValueError("Cache failed to initialize. Ensure 'use_cache=True' is supported.")

        # Get the baseline logits from the last position of the original sequence
        with torch.no_grad():
            baseline_out = self.model(**inputs, use_cache=True)
        baseline_logits = baseline_out.logits[:, -1, :]
        
        influence_scores = []
        
        # Iterate specifically through ssm_states
        for i in range(len(cache.ssm_states)):
            target_ssm = cache.ssm_states[i]
            
            # Skip attention layers (empty states have 0 dimension in ssm_states)
            if target_ssm.numel() == 0:
                influence_scores.append(0.0)
                continue
                
            # Create a deep copy to isolate perturbations
            test_cache = copy.deepcopy(cache)
            
            # Nudge ONLY this specific layer
            perturbation = torch.randn_like(test_cache.ssm_states[i]) * test_strength
            test_cache.ssm_states[i].add_(perturbation)
            
            with torch.no_grad():
                # Pass modified cache back; inputs should only be the last token or prompt
                steered_out = self.model(
                    input_ids=inputs.input_ids, 
                    past_key_values=test_cache,
                    use_cache=True
                )
            
            divergence = torch.nn.functional.mse_loss(
                steered_out.logits[:, -1, :], 
                baseline_logits
            ).item()
            
            influence_scores.append(divergence)
            print(f"Layer {i} (SSM) Influence: {divergence:.6f}")

        return influence_scores


if __name__ == "__main__":
    steerer = GraniteSteerer()
    
    prompt = "Why did the United States declare independence from Britain?"
    
    print("\nUnsteered output:")
    print(steerer.generate_unsteered(prompt))
    
    print("\nSteered output:")
    target_layers = [36, 37, 38]
    print(steerer.generate_steered(prompt, perturb_strength=0.1, target_layers=target_layers))
    
    # Cleanup
    if 'steerer' in globals():
        del steerer.model
        del steerer.tokenizer
        steerer = None
        torch.cuda.empty_cache()
        print("Model unloaded and GPU cache cleared.")
