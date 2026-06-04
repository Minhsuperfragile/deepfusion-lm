import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from model import DeepfusionConfig
from dotenv import load_dotenv

load_dotenv()

# 1. Configuration
corpus_path = os.getenv("CORPUS_PATH", "G:\\hf\\vietnamese-book")
tokenizer_path = os.getenv("TOKENIZER_JSON_PATH", "DeepfusionLM_tokenizer.json")
output_path = os.getenv("PROCESSED_PRETRAIN_DATA_PATH", "./tokenized_vietnamese_book")

# Load config to get the correct sequence length
config = DeepfusionConfig()
context_length = config.max_seq_len

if os.path.exists(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        model_max_length=config.max_seq_len
    )
    # Ensure special tokens are recognized by the transformers wrapper
    tokenizer.cls_token = "[BOS]"
    tokenizer.pad_token = "[EOS]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "[START_ID]",
        "[END_ID]",
        "[EOT]"
    ]
})
else:
    print(f"Error: {tokenizer_path} not found. Please run train_tokenizer.py first.")

start_token = "[START_ID]"
end_token = "[END_ID]"
eot_token = "[EOT]"

### Chat Template for SFT ###
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ bos_token if loop.first else '' }}"
    f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
    "{{ message['content'] }}"
    f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    f"{{{{ '{start_token}' + 'assistant' + '{end_token}' }}}}"
    "{% endif %}"
)

# Instead of saving just the JSON file
tokenizer.save_pretrained(os.getenv("TOKENIZER_PATH", "./DeepfusionLM_tokenizer"))