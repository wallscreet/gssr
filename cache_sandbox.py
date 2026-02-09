import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class CFG:
    MAX_NEW_TOKENS=120
    TEMPERATURE=0.8
    TOP_P=0.9
    DO_SAMPLE=True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "ibm-granite/granite-4.0-h-350m"
    D_TYPE = torch.bfloat16

class LocalModel:
    def __init__(self):
        print("Loading tokenizer and model...")
        self.cfg = CFG()
        self.device = self.cfg.DEVICE
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_PATH)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.cfg.MODEL_PATH,
            device_map=self.device,
            dtype=self.cfg.D_TYPE,
        )
        self.model.eval()
    
    def _prepare_inputs(self, prompt):
        chat = [{"role": "user", "content": prompt}]
        
        chat_text = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return self.tokenizer(chat_text, return_tensors="pt").to(self.device)
    
    def generate(self, prompt):
        inputs = self._prepare_inputs(prompt=prompt)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
        )
        
        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    def summarize_and_capture_state(self, summary_prompt):
        """
        Run a summary/knowledge prompt through the model and capture the final SSM state.
        """
        inputs = self._prepare_inputs(summary_prompt)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=100,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
        cache = out.past_key_values
        
        return cache

    def generate_using_cache(self, prompt, cached_state, max_new_tokens=100):
        """
        Generate continuation using a pre-cached state (prompt caching / state injection).
        """
        inputs = self._prepare_inputs(prompt)
        input_ids = inputs.input_ids  # [1, prompt_len]
        
        # Safety: ensure input is not empty
        if input_ids.shape[1] == 0:
            print("Warning: empty tokenized prompt - adding BOS token")
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
        
        # Start with the provided cached state
        past_key_values = cached_state
        
        # We'll collect generated token IDs (excluding prompt)
        generated_tokens = []
        
        # First forward pass: process the new prompt using the pre-cached state
        # This updates the cache with prompt information
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        
        # Update cache after prompt processing
        past_key_values = outputs.past_key_values
        
        # autoregressive generation loop
        current_ids = input_ids  # keep full sequence so far (needed for some models)
        
        for _ in range(max_new_tokens):
            # Get logits for the last token position
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            
            if self.cfg.DO_SAMPLE:
                probs = torch.softmax(logits / self.cfg.TEMPERATURE, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.item())
            
            # Append to current sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Next forward pass: only the new token, with updated cache
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
        
        # Decode newly generated  tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text, past_key_values


# Module Scripts
# ---------------------------------------------------------

corpus = """
Granite models are hybrid SSM-transformer models designed for efficient reasoning and generation. Smaller models can retain information in cache across generations, enabling efficient prompt reuse. For our purposes, the ability to retain cache is the primary way Granite models enable efficient inference.
"""

question = "How do Granite models enable efficient inference?"


def summarize_corpus():
    _prompt = f"Generate a detailed summary of the following information:\n{corpus}"
    model = LocalModel()
    
    print("\n--- Generating corpus summary ---")
    summary_response = model.generate(prompt = _prompt)

    print("\nSummary output:\n", summary_response)


def bare_response():
    model = LocalModel()
    _prompt = question
    
    print("\n--- Generating Response ---")
    response = model.generate(prompt = _prompt)
    
    print(f"\nModel Response:\n{response}")


def prompt_inject_response():
    model = LocalModel()
    _prompt = f"\nReference Info:\n{corpus}\nQuestion:\n{question}"
    
    response = model.generate(prompt=_prompt)
    print(f"\nModel Response:\n{response}")

def cached_state_response():
    model = LocalModel()
    _summary_prompt = f"Generate a detailed summary of the following information:\n{corpus}"
    _prompt = f"{question}"
    
    cached_state = model.summarize_and_capture_state(summary_prompt=_summary_prompt)
    response, _ = model.generate_using_cache(prompt=_prompt, cached_state=cached_state)
    print(f"Response:\n{response}")

if __name__ == "__main__":
    # summarize_corpus()
    # bare_response()
    # prompt_inject_response()
    cached_state_response()