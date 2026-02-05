# Hacking Granite-4.0 Hybrid

## Finding and Extracting the Mamba-2 Recurrent States

IBM's Granite-4.0 series uses a hybrid architecture that combines traditional attention layers with Mamba-2 (state-space model) layers for efficiency and long-context handling. My goal was to reach inside the model during inference, locate the recurrent SSM states (the "memory" that Mamba-2 carries forward), and eventually perturb them for steering experiments.

To do this, I wrote a short script that:

1. Loads the model and tokenizer  
2. Prepares a simple chat prompt  
3. Registers forward hooks on all Mamba-related submodules  
4. Runs a forward pass and short generation to trigger caching  
5. Inspects the resulting cache object to find where the recurrent states are stored

Here's the core code (slightly cleaned for clarity):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_path = "ibm-granite/granite-4.0-h-1b"
# model_path = "ibm-granite/granite-4.0-h-micro"
# model_path = "ibm-granite/granite-4.0-h-350m"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
model.eval()

# Simple prompt
chat = [{"role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location."}]
chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_text, return_tensors="pt").to(device)

# Storage for captured states
ssm_states = []

def capture_mamba_state(module, input, output):
    if isinstance(output, tuple) and len(output) > 1:
        state = output[1]
        if isinstance(state, torch.Tensor):
            ssm_states.append(state.detach().cpu())
            print(f"Captured tuple from {module.__class__.__name__}: shape {state.shape}")

# Register hooks on Mamba-related submodules
for name, module in model.named_modules():
    if hasattr(module, 'register_forward_hook'):
        if any(kw in name.lower() for kw in ['mamba', 'conv1d', 'in_proj', 'ssm']):
            module.register_forward_hook(capture_mamba_state)

# Run forward pass + short generation to initialize cache
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, use_cache=True)

with torch.no_grad():
    gen_outputs = model.generate(**inputs, max_new_tokens=5, use_cache=True, return_dict_in_generate=True)

# Inspect the cache
cache = gen_outputs.past_key_values
print(f"Cache type: {type(cache).__name__}")

# Look for ssm_states attribute
if hasattr(cache, 'ssm_states'):
    ssm_list = cache.ssm_states
    print(f"Found ssm_states list with {len(ssm_list)} entries")
    for i, s in enumerate(ssm_list):
        if isinstance(s, torch.Tensor):
            print(f"  State {i}: shape={s.shape}, mean={s.mean().item():.4f}")
```

### What I Found

After running the forward pass and a short generation step, the model returns a custom cache object called `HybridMambaAttentionDynamicCache`.

Inside it is an attribute `ssm_states`, a list with one state per decoder layer.

Each entry is a tensor representing the recurrent state for that layer.

---

#### IBM-granite/granite-4.0-h-1b

- **Mamba-2 layers** (36 of them): shape `[1, 48, 64, 128]`  
  → Batch 1 × 48 groups/heads × 64 inner dimension × 128 core state dimension
- **attention layers** (4 of them): shape `[1, 0]` (empty placeholder) with mean `nan`

The attention layers are at index position 5, 15, 25, 35.

---

#### IBM-granite/granite-4.0-h-350m

- **Mamba-2 layers** (28 of them): shape `[1, 48, 32, 128]`
  → Batch 1 x 48 groups/heads x 32 inner dimension x 128 core state dimension
- **attention layers** (4 of them): shape `[1, 0]` (empty placeholder) with mean `nan`

The attention layers are at index positions 10, 13, 17, 27.

---

This means the recurrent memory is only carried by the Mamba-2 layers, and the hybrid design keeps a consistent slot for every layer in the cache, even if some are empty.

## Why This Matters

Being able to reliably locate and extract these SSM states during inference is the foundation for my overall goal: perturbing them with a goal-conditioned vector to steer the model's output trajectory.

The fact that the states are exposed in a structured cache object (rather than hidden internals) makes real-time modification feasible which is exactly what I need for latent-space steering experiments.

## Perturbing the late Mamba States and Observing Output Changes

Once I knew where the recurrent SSM states lived, the obvious next question was:
**Can I reach in and change them mid-generation to alter the model’s output?**

I wrote a class called `GraniteSteerer` to test exactly that. My idea was:

1. Run normal (unsteered) generation as a baseline.
2. Run steered generation where, at every step, I take selected SSM states, apply a small random perturbation, and let the model continue with the modified cache.

The code does this in a step-by-step generation loop (instead of `model.generate()`), so I can modify the cache between tokens.

### Experiment Code

```python
class GraniteSteerer:
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.model.eval()
        self.device = device
        self.d_state = 128  # core SSM state dimension

    def generate_unsteered(self, prompt, max_new_tokens=100):
        inputs = self._prepare_inputs(prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    def generate_steered(self, prompt, max_new_tokens=100, perturb_strength=0.05, target_layers=range(36, 40)):
        inputs = self._prepare_inputs(prompt)
        
        # Initialize cache with a tiny generation step
        init_out = self.model.generate(**inputs, max_new_tokens=1, use_cache=True, return_dict_in_generate=True)
        current_cache = init_out.past_key_values
        generated_ids = inputs.input_ids.clone()

        # Random goal vector (placeholder to be replaced with meaningful goal)
        goal = torch.randn(1, self.d_state, device=self.device) * perturb_strength

        for step in range(max_new_tokens):
            step_out = self.model(
                input_ids=generated_ids[:, -1:],
                past_key_values=current_cache,
                use_cache=True,
                return_dict=True,
            )
            
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            current_cache = step_out.past_key_values
            
            # Perturb selected late SSM states
            if hasattr(current_cache, 'ssm_states'):
                ssm_list = current_cache.ssm_states
                for layer_idx in target_layers:
                    if layer_idx >= len(ssm_list):
                        continue
                    target_ssm = ssm_list[layer_idx]
                    
                    # Flatten safely to [1, 128]
                    if target_ssm.dim() >= 3:
                        current_flat = target_ssm.mean(dim=list(range(1, target_ssm.dim()-1)))
                    else:
                        current_flat = target_ssm[:, -self.d_state:] if target_ssm.shape[-1] > self.d_state else target_ssm
                    
                    # Simple perturbation: directional nudge toward goal
                    perturb_vec = torch.tanh(goal - current_flat)
                    perturb_strength_val = torch.sigmoid(current_flat.abs().mean(dim=-1))
                    perturbed_flat = current_flat + perturb_strength_val.unsqueeze(-1) * perturb_vec * perturb_strength
                    
                    # Broadcast back and apply
                    perturbation = perturbed_flat.view(1, *(1,) * (len(target_ssm.shape)-2), self.d_state).expand_as(target_ssm)
                    target_ssm.add_(perturbation)
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

#### What Happened When I Ran It

- **Unsteered output** (normal generation): Clean, short, correct answers.

```bash
Unsteered output:
systemYou are a helpful assistant. Please ensure responses are professional, accurate, and safe.
userWhy did the United States declare independence from Britain?
assistantThe United States declared independence from Britain primarily due to a series of perceived injustices and infringements on the rights of the American colonies by the British government. The main reasons include:

1. Taxation without Representation: The British Parliament passed several taxes and
```

- **Steered output** (with perturbation on layers 36–39):  
  - At low strength (~0.03–0.05): subtle changes or early collapse into gibberish/special tokens.  
  - At higher strength (0.1–0.2): longer initial coherence, then sudden lock-in to strong repetition (e.g., endless "248248..." strings).

```bash
# perturb_strenth=0.01, target_layers=range(36, 40)

Steered output:
systemYou are a helpful assistant. Please ensure responses are professional, accurate, and safe.
userWhy did the United States declare independence from Britain?
assistantThe United States declared independence from Britain in **<|fim_middle|> Construct052729fad …

2����������������������������� excer[
```

```bash
# perturb_strenth=0.2, target_layers=range(36, 40)

Steered output:
systemYou are a helpful assistant. Please ensure responses are professional, accurate, and safe.
userWhy did the United States declare independence from Britain?
assistantThe United Statesmodxarts assistir.CompareTag  BIT     BIT     BIT     BIT     BIT     BIT     BIT    BIT      BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT     BIT    BIT      BIT     BIT     BIT     BIT @Module     BITaires<|fim_middle|>about.Disclaimer @Module 899      BIT    BIT      BIT     BITcontrolled
```

```bash
# perturb_strength=0.05, target_layers=range(36, 40)

Steered output:
systemYou are a helpful assistant. Please ensure responses are professional, accurate, and safe.
userWhy did the United States declare independence from Britain?
assistantThe United Statesariangnore.CompareTag.CompareTag.CompareTag.CompareTag adolescente712.CompareTag<|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|>erti\_<|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|>loo<|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|><|fim_middle|>
```

```python
# perturb_strength=0.50, target_layers=range(36, 40)

Steered output:
systemYou are a helpful assistant. Please ensure responses are professional, accurate, and safe.
userWhy did the United States declare independence from Britain?
assistantThe United.EventSystems3optgroup<|fim_pad|><|fim_suffix|>@include      CStringionales051<|fim_suffix|>927;"><? consumed<|fim_suffix|>ं follando&id**)&






pintra_inicio_inicio8\$3 diseñador;"><?<|fim_suffix|>ं135 . им<|fim_middle|> //</\ \
```

The repetition collapse is classic SSM behavior when the recurrent state is pushed far enough off its stable path, the model gets stuck in a low-probability loop reinforced by the recurrence. I did find the results at higher pertuber strengh (.5) more interesting. It seems that at extreme perturbation, the SSM state no longer aligns with the English token manifold and falls back to whatever tokens have even slightly higher probability under the scrambled distribution and multilingual / code / markup tokens seem to be more common in the long tail of the model's vocabulary.

### Why This Was a Success

Even though the steered outputs often ended in nonsense, the experiment proved the core idea:  
**We can reach into the Mamba-2 recurrent states during generation, modify them on the fly, and see immediate changes in the output trajectory.**

Random perturbation already causes visible steering, sometimes subtle drift and sometimes catastrophic collapse. This confirms the mechanism does in fact interact with and influence the outputs of the model.

The next step is to replace the random vector with a **goal-conditioned** one (e.g., derived from a desired continuation or answer) and tune the strength / gating so the steering is meaningful and stable instead of destructive.
