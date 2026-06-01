import os
from datasets import load_dataset
from transformers import AutoTokenizer
from model import DeepfusionConfig
from tokenizers.processors import TemplateProcessing
from dotenv import load_dotenv
load_dotenv()

# 1. Configuration
corpus_path = os.getenv("CORPUS_PATH", r"G:\hf\vietnamese-book")
tokenizer_path = os.getenv("TOKENIZER_PATH", "./DeepfusionLM_tokenizer")
output_path = os.getenv("PROCESSED_PRETRAIN_DATA_PATH", "./tokenized_vietnamese_book")

# Load config to get the correct sequence length
config = DeepfusionConfig()
context_length = config.max_seq_len

def prepare_tokenized_data():
    # 2. Load the dataset from the local folder
    # The 'text' loader automatically finds all .txt files in the directory
    print(f"Loading dataset from {corpus_path}...")
    if not os.path.exists(corpus_path):
        print(f"Error: Folder {corpus_path} not found.")
        return

    dataset = load_dataset("text", data_dir=corpus_path, split="train")

    # 3. Load your custom-trained tokenizer
    # We use PreTrainedTokenizerFast to load the .json file from train_tokenizer.py
    print(f"Loading tokenizer from {tokenizer_path}...")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Ensure special tokens are recognized by the transformers wrapper
        tokenizer.pad_token = "[EOS]"
        tokenizer.mask_token = "[MASK]"
        tokenizer.unk_token = "[UNK]"
        tokenizer.bos_token = "[BOS]"
        tokenizer.eos_token = "[EOS]"

        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single = f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
            special_tokens = [
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id)
            ]
        )

    else:
        print(f"Error: {tokenizer_path} not found. Please run train_tokenizer.py first.")
        return

    # 4. Tokenization and Chunking function
    # This function tokenizes text and groups them into chunks of context_length
    def tokenize_and_chunk(examples):
        # Tokenize the batch of text
        outputs = tokenizer(examples["text"], truncation=False, padding=False)
        
        # Concatenate all token IDs in this batch
        all_ids = []
        for ids in outputs["input_ids"]:
            if len(ids) == 0:
                continue
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id) # Add EOS between documents
        
        # Create fixed-length chunks
        total_len = len(all_ids)
        result_chunks = []
        for i in range(0, total_len, context_length):
            chunk = all_ids[i : i + context_length]
            if len(chunk) == context_length:
                result_chunks.append(chunk)
                
        return {"input_ids": result_chunks}

    # 5. Process the dataset
    print(f"Tokenizing and chunking dataset (context_length={context_length})...")
    tokenized_dataset = dataset.map(
        tokenize_and_chunk, 
        batched=True, 
        batch_size=1000,
        num_proc=16, # Use multiple CPU cores
        remove_columns=dataset.column_names
    )

    # 6. Save the processed dataset to disk
    print(f"Saving tokenized dataset to {output_path}...")
    tokenized_dataset.save_to_disk(output_path)
    print("Done! You can now use this dataset in your pretraining script.")

if __name__ == "__main__":
    prepare_tokenized_data()
