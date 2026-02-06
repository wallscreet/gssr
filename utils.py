import torch

def debug_module(module, name: str = "", enabled: bool = False):
    """Print shape, device, and dtype of a module's parameters."""
    if not enabled:
        return
    
    print(f"\nDEBUG: {name or module.__class__.__name__}")
    print(f"  - Type: {module.__class__.__name__}")
    
    # Weight (most modules have it)
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"  - Weight shape: {module.weight.shape}")
        print(f"  - Weight device: {module.weight.device}")
        print(f"  - Weight dtype: {module.weight.dtype}")
    else:
        print("  - No weight attribute or None")
    
    # Bias - safe check
    if hasattr(module, 'bias'):
        bias = module.bias
        if bias is not None:
            if isinstance(bias, torch.Tensor):
                print(f"  - Bias shape: {bias.shape}")
                print(f"  - Bias device: {bias.device}")
                print(f"  - Bias dtype: {bias.dtype}")
            else:
                print(f"  - Bias exists but not a tensor: {type(bias)}")
        else:
            print("  - Bias is None")
    else:
        print("  - No bias attribute")
    
    # General param info (first param if exists)
    params = list(module.parameters())
    if params:
        first_param = params[0]
        print(f"  - First param device: {first_param.device}")
        print(f"  - First param dtype: {first_param.dtype}")
        print(f"  - Number of params: {len(params)}")
    else:
        print("  - No parameters")


def debug_tensor(t, name=""):
    if t is None:
        print(f"DEBUG tensor {name}: None")
        return
    
    shape = t.shape
    device = t.device
    dtype = t.dtype
    
    if t.numel() > 0:
        mean_val = t.mean().item()
        mean_str = f"{mean_val:.6f}"
    else:
        mean_str = "empty"
    
    print(f"DEBUG tensor {name}: shape={shape}, device={device}, dtype={dtype}, mean={mean_str}")