import os
import torch
from model import DeepfusionConfig, DeepfusionLM
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

# 1. Load the tokenizer
tokenizer_json_path = "DeepfusionLM_tokenizer.json"
if not os.path.exists(tokenizer_json_path):
    tokenizer_json_path = "DeepfusionLM_tokenizer/tokenizer.json"

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_json_path,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[EOS]",
    mask_token="[MASK]"
)

# 2. Instantiate configuration and model skeleton
config = DeepfusionConfig()
config.vocab_size = len(tokenizer) 
model = DeepfusionLM(config)

# 3. Load the saved weights from your custom model's checkpoint directory
# (Note: Loading the ModernBERT checkpoint into DeepfusionLM will cause key mismatch errors)
checkpoint_path = "my_experiment/final_model/model.safetensors"

if not os.path.exists(checkpoint_path):
    print(f"Warning: Checkpoint not found at '{checkpoint_path}'.")
    print("Please run 'pretrain_my_model.py' first to train and save your model.")
    print("Using a randomly initialized model for demonstration...")
else:
    # 4. Load the state dict into your model
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    print(f"Custom model loaded successfully from '{checkpoint_path}'!")

# 5. Move to GPU if available and set to evaluation mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Define the text with a [MASK] token
prompt = "Ngày xửa ngày xưa, ở một ngôi làng nọ có một [MASK] xinh đẹp."

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# Find mask token index
mask_token_id = tokenizer.mask_token_id
mask_token_index = torch.where(input_ids == mask_token_id)[1]

if len(mask_token_index) == 0:
    raise ValueError(f"No mask token ({tokenizer.mask_token}) found in the prompt!")

# Construct attention mask (False means attend in SDPA for this model)
batch_size, seq_len = input_ids.shape
attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

# Run inference
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask)

# Retrieve logits at the mask position
mask_token_logits = logits[0, mask_token_index, :]

# Get top 5 predicted token IDs
top_k = 5
top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1).indices[0].tolist()

# Print results
print(f"\nPrompt: {prompt}")
print("-" * 40)
print("Top MLM Predictions:")
for i, token_id in enumerate(top_k_tokens):
    predicted_token = tokenizer.decode([token_id])
    print(f" {i + 1}: {predicted_token.strip()}")
