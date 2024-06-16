import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from lit_gpt.model import GPT, Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id='nampdn-ai/samba-chattrust-421m-v2', filename='iter-017200-ckpt.pth')
print(model_path)
# Configuration
model_name = "Samba_421M"
train_config = "tsz512x4k_20B"
name = train_config + "_" + model_name
# model_path = Path("/work/trainer/Samba/out/iter-003200-ckpt.pth") # Update with the path to your saved model
tokenizer = AutoTokenizer.from_pretrained("nampdn-ai/tknz-20k", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# Function to load the trained model
def load_model(model_path: Path):
    config = Config.from_name(model_name)
    model = GPT(config)
    state_dict = torch.load(model_path)["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded!")
    
    model.eval()
    print("Device:", device)
    return model

def generate_text(model, tokenizer, initial_text, max_new_tokens, temperature=0.3, top_k=20, top_p=0.95, repeat_penalty=1.2):
    model.eval()
    device = next(model.parameters()).device  # Ensures that tensors are on the same device as model
    input_ids = tokenizer(initial_text, return_tensors='pt').input_ids.to(device)
    print("Inputs:", input_ids)
    generated_ids = input_ids.clone()

    past_tokens = set()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits /= temperature

            # Apply repeat penalty
            for token_id in past_tokens:
                next_token_logits[0, token_id] /= repeat_penalty

            # Apply top-k and top-p filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k).values.min()
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')

            next_token_id = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

            # Store sampled tokens to apply repeat penalty in future iterations
            past_tokens.add(next_token_id.item())

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

model = load_model(model_path)

# Example usage
initial_text = """### Problem:
To make pizza, together with other ingredients, Kimber needs 10 cups of water, 16 cups of flour, and 1/2 times as many teaspoons of salt as the number of cups of flour. Calculate the combined total number of cups of water, flour, and teaspoons of salt that she needs to make the pizza.

### Solution:"""

initial_text = """### Problem:
What is the meaning of the saying “Christ is King”?

### Solution:"""


initial_text = """### Problem:
What does it mean to be baptized into Christ (Galatians 3:27)?

### Solution:"""

initial_text = """### Problem:
How long did it take Noah to build the ark?

### Solution:"""


initial_text = """### Problem:
What does it mean to be seasoned with salt (Mark 9:49)?

### Solution:"""

initial_text = """Vào ngày sáng tạo cuối cùng Đức Chúa Trời nói:"""

initial_text = """### Problem:
When were Adam and Eve created?

### Solution:"""


generated_text = generate_text(model, tokenizer, initial_text, max_new_tokens=512)
print("Generated Text:", generated_text)