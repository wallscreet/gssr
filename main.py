import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
#model_path = "ibm-granite/granite-4.0-h-1b"
model_path = "ibm-granite/granite-4.0-h-350m"
# model_path = "ibm-granite/granite-4.0-h-micro"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    dtype=torch.bfloat16,
)
model.eval()

# Input
chat = [
    {"role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location."},
]
chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_text, return_tensors="pt").to(device)

print(f"Input shape: {inputs.input_ids.shape}")

# ==============================================
# captured SSM/recurrent states
# ==============================================
ssm_states = []

def capture_mamba_state(module, input, output):
    captured = False
    
    # Check if submodule returned a tuple with a potential state
    if isinstance(output, tuple) and len(output) > 1:
        potential_state = output[1]
        if isinstance(potential_state, torch.Tensor):
            ssm_states.append(potential_state.detach().cpu())
            print(f"→ Captured TUPLE from {module.__class__.__name__} "
                  f"(shape: {potential_state.shape}, mean: {potential_state.mean().item():.4f})")
            captured = True
    
    # Check common attribute names where state might be stored
    for attr in ['h', 'state', 'recurrent_state', 'ssm_state', 'hidden_state']:
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, torch.Tensor):
                ssm_states.append(val.detach().cpu())
                print(f"→ Captured ATTR '{attr}' from {module.__class__.__name__} "
                      f"(shape: {val.shape}, mean: {val.mean().item():.4f})")
                captured = True

# ==============================================
# Register hooks on all relevant submodules
# ==============================================
print("Registering hooks...")
registered = 0
for name, module in model.named_modules():
    if hasattr(module, 'register_forward_hook'):
        if any(kw in name.lower() for kw in ['mamba', 'conv1d', 'in_proj', 'ssm']):
            try:
                module.register_forward_hook(capture_mamba_state)
                registered += 1
                # print(f"Registered hook on: {name}")  # uncomment if you want full list again
            except Exception as e:
                print(f"Failed to register on {name}: {e}")

print(f"Registered {registered} hooks")

# ==============================================
# Run forward pass
# ==============================================
print("Running forward pass...")
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=True,
    )

# ==============================================

# ==============================================
print("\nRunning short generation to trigger cache...")
with torch.no_grad():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=5, # just enough to initialize cache
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=False,
    )

print("Generation completed. Checking cache...")

if hasattr(gen_outputs, 'past_key_values') and gen_outputs.past_key_values is not None:
    cache = gen_outputs.past_key_values
    print(f"past_key_values type: {type(cache)}")
    if isinstance(cache, (list, tuple)):
        print(f"Number of layers cached: {len(cache)}")
        for layer_idx, layer_cache in enumerate(cache[:8]):  # Inspect first 8
            print(f"  Layer {layer_idx}:")
            if isinstance(layer_cache, tuple):
                for sub_idx, sub_item in enumerate(layer_cache):
                    if isinstance(sub_item, torch.Tensor):
                        print(f"    sub-{sub_idx}: shape={sub_item.shape}, "
                              f"dtype={sub_item.dtype}, mean={sub_item.mean().item():.4f}")
                    else:
                        print(f"    sub-{sub_idx}: {type(sub_item)}")
            elif isinstance(layer_cache, torch.Tensor):
                print(f"    direct tensor: shape={layer_cache.shape}, "
                      f"mean={layer_cache.mean().item():.4f}")
            else:
                print(f"    other type: {type(layer_cache)}")
    else:
        print("Cache is not a list/tuple — inspect manually:", cache)
else:
    print("No past_key_values even after generation.")


print("\nInspecting HybridMambaAttentionDynamicCache attributes...")
cache = gen_outputs.past_key_values

# Print public attributes/methods
print("Cache attributes:")
for attr in dir(cache):
    if not attr.startswith('__'):
        try:
            val = getattr(cache, attr)
            print(f"  .{attr}: type={type(val)}")
            if isinstance(val, (list, tuple)) and len(val) > 0:
                print(f"    length={len(val)}, first item type={type(val[0])}")
                if isinstance(val[0], torch.Tensor):
                    print(f"      example shape={val[0].shape}, mean={val[0].mean().item():.4f}")
            elif isinstance(val, torch.Tensor):
                print(f"    tensor shape={val.shape}, mean={val.mean().item():.4f}")
            elif isinstance(val, dict):
                print(f"    dict keys={list(val.keys())}")
        except Exception as e:
            print(f"  .{attr}: <error accessing: {e}>")

# Try patterns for hybrid caches
try:
    if hasattr(cache, 'mamba_states') or hasattr(cache, 'ssm_states'):
        states_attr = 'mamba_states' if hasattr(cache, 'mamba_states') else 'ssm_states'
        mamba_states = getattr(cache, states_attr)
        print(f"\nFound '{states_attr}': type={type(mamba_states)}, length={len(mamba_states) if hasattr(mamba_states, '__len__') else 'N/A'}")
        if isinstance(mamba_states, (list, tuple)) and len(mamba_states) > 0:
            for i, s in enumerate(mamba_states):
                if isinstance(s, torch.Tensor):
                    print(f"  Mamba state {i}: shape={s.shape}, mean={s.mean().item():.4f}")
except:
    pass

# try:
#     print("\nTrying cache[0] access...")
#     layer0_cache = cache[0]
#     print(f"cache[0] type: {type(layer0_cache)}")
#     if isinstance(layer0_cache, tuple):
#         for j, sub in enumerate(layer0_cache):
#             if isinstance(sub, torch.Tensor):
#                 print(f"  sub-{j}: shape={sub.shape}, mean={sub.mean().item():.4f}")
# except Exception as e:
#     print(f"cache[0] failed: {e}")


# print("\nExtracting SSM states from cache...")
# cache = gen_outputs.past_key_values
# if hasattr(cache, 'ssm_states'):
#     ssm_list = cache.ssm_states
#     print(f"Total SSM entries: {len(ssm_list)}")
    
#     last_ssm = ssm_list[-1]
#     print(f"Last SSM state shape: {last_ssm.shape}")
#     print(f"Mean: {last_ssm.mean().item():.4f}")
    
#     flattened = last_ssm.mean(dim=[1,2])
#     print(f"Flattened example: shape={flattened.shape}, mean={flattened.mean().item():.4f}")
