from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_tokenizer(model_name="answerdotai/ModernBERT-base",
                  bos_token="<BOS>",
                  eos_token="<EOS>",
                  start_token="<START_ID>",
                  end_token="<END_ID>",
                  eot_token="<EOT_ID>"):
    """
    Load tokenizer, add special tokens, and set up a chat template.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Define our special tokens
    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "additional_special_tokens": [
            start_token,
            end_token,
            eot_token,
        ]
    }

    # Add them to tokenizer
    tokenizer.add_special_tokens(special_tokens)

    # Set EOS token as PAD token and CLS to BOS Token
    tokenizer.pad_token = eos_token
    tokenizer.cls_token = bos_token
    
    ### Template Processing for Tokenizing Pretraining Data ###
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        special_tokens=[
            (bos_token, tokenizer.bos_token_id),
            (eos_token, tokenizer.eos_token_id),
        ],
    )

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

    return tokenizer


def test_tokenizer():
    print("=== Pretraining Mode Test ===")
    tok_pre = get_tokenizer(mode="pretrain")
    text = "Hello world"
    ids = tok_pre(text, padding=True, return_tensors="pt")["input_ids"][0]
    decoded = tok_pre.decode(ids, skip_special_tokens=False)
    print("Text:", text)
    print("Encoded IDs:", ids.tolist())
    print("Decoded:", decoded)
    print()

    print("=== SFT Mode Test ===")
    tok_sft = get_tokenizer(mode="sft")
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."}
    ]
    encoded = tok_sft.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
    decoded = tok_sft.decode(encoded, skip_special_tokens=False)
    print("Messages:", messages)
    print("Encoded IDs:", encoded)
    print("Decoded:", decoded)
    print()

    print("=== QA Mode Test ===")
    messages = [
        {"role": "user", "content": "Who wrote Hamlet?"}
    ]

    # Add assistant prompt
    encoded = tok_sft.apply_chat_template(
        messages,
        tokenize=True,
        add_special_tokens=True,
        add_generation_prompt=True
    )

    print("Decoded:", tok_sft.decode(encoded, skip_special_tokens=False))

# Run tests
if __name__ == "__main__":
    test_tokenizer()