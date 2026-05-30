# pyrefly: ignore [missing-import]
from datasets import load_dataset, load_from_disk
import time
from transformers import PreTrainedTokenizerFast
import argparse
import os

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "DeepfusionLM_tokenizer")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "vietnamese-history-qa")


parser = argparse.ArgumentParser(description="Finemath Data Prep")

parser.add_argument(
    "--test_split_pct", 
    default=0.005, 
    help="What percent of data do you want to use for Train/Test split",
    type=float
)

parser.add_argument(
    "--context_length", 
    default=1024, 
    help="Pass in argument to override the default in Config, but then make sure config \
        reflects this when training",
    type=int
)

parser.add_argument(
    "--path_to_data_store", 
    required=True, 
    help="Path to where you want to save the final tokenized dataset",
    type=str
)

parser.add_argument(
    "--huggingface_cache_dir",
    default=None,
    help="path to huggingface cache directory if different from default",
    type=str
)

parser.add_argument(
    "--dataset_split_seed",
    default=42, 
    help="Seed to ensure reproducible split of dataset",
    type=int
)

parser.add_argument(
    "--num_workers",
    default=16, 
    help="Number of workers you want to use to process dataset",
    type=int
)

parser.add_argument(
    "--hf_model_name",
    default="answerdotai/ModernBERT-base",
    help="Name of model so we can use the huggingface tokenizer", 
    type=str
)

parser.add_argument(
    "--large_dataset",
    action="store_true"
)

parser.add_argument(
    "--batch_size",
    type=int, 
    default=100
)


def get_tokenizer():
    """
    Load the DeepfusionLM tokenizer từ local directory.
    Special tokens (đã được lưu sẵn trong tokenizer):
      - bos_token  : [BOS]
      - eos_token  : [EOS]
      - pad_token  : [EOS]
      - cls_token  : [BOS]
      - unk_token  : [UNK]
      - mask_token : [MASK]
      - additional : [START_ID], [END_ID], [EOT]
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    # Gán lại pad & cls token (runtime, không lưu trong config)
    tokenizer.pad_token = "[EOS]"
    tokenizer.cls_token = "[BOS]"

    return tokenizer


def format_final_content(content):
    """
    Serialize content dict của assistant 'final' thành plain text.
    Bỏ hết phần thinking (channel='analysis' đã bị lọc trước đó).
    """
    if isinstance(content, str):
        return content
    parts = []
    if "title" in content:
        parts.append(f"**{content['title']}**")
    if "time" in content:
        parts.append(f"Thời gian: {content['time']}")
    if "background" in content:
        parts.append(f"Bối cảnh: {content['background']}")
    if "event" in content:
        parts.append(f"Sự kiện: {content['event']}")
    if "meaning" in content:
        meanings = content["meaning"]
        meaning_str = "\n".join(f"- {m}" for m in meanings) if isinstance(meanings, list) else meanings
        parts.append(f"Ý nghĩa:\n{meaning_str}")
    if "historical_value" in content:
        parts.append(f"Giá trị lịch sử: {content['historical_value']}")
    return "\n".join(parts)


def process_sample(sample, tokenizer, context_length):
    """
    Xử lý 1 sample:
      1. Bỏ messages có channel='analysis' (chứa thinking)
      2. Giữ: system, user, assistant channel='final'
      3. Serialize content dict → string
      4. Apply chat template + tokenize
    """
    filtered = []
    for msg in sample["messages"]:
        # Bỏ analysis channel (thinking)
        if msg.get("channel") == "analysis":
            continue
        role = msg["role"]
        content = msg["content"]
        # Serialize content dict (assistant 'final') thành string
        if isinstance(content, dict):
            content = format_final_content(content)
        filtered.append({"role": role, "content": content})

    result = tokenizer.apply_chat_template(
        filtered,
        tokenize=True,
        truncation=False,      # không cắt — filter sau
        return_tensors=None,   # trả về list[int]
    )
    # apply_chat_template có thể trả về BatchEncoding (UserDict, KHÔNG phải dict thuần)
    # hoặc list[int]. KHÔNG dùng isinstance(result, dict) vì BatchEncoding kế thừa
    # UserDict → isinstance(result, dict) trả False → bỏ qua unwrap → bug.
    # Thay vào đó: nếu không phải list → extract input_ids qua attribute hoặc key.
    if isinstance(result, list):
        input_ids = result
    elif hasattr(result, "input_ids"):
        input_ids = result.input_ids     # BatchEncoding attribute access
    else:
        input_ids = result["input_ids"]  # fallback key access

    # Đảm bảo là flat list[int], không phải nested list/tensor
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]   # unwrap batch dimension nếu có

    return {"input_ids": [int(t) for t in input_ids]}


def get_answer_mask(example, tokenizer):
    """
    Tạo query_mask: 0 = prompt (system/user/role headers), 1 = answer (assistant content).
    Logic: đếm tổng số [END_ID] token trong sequence, rồi
    bắt đầu mask=1 NGAY SAU [END_ID] cuối cùng — tức sau role header của assistant.
    Ví dụ với 3 turns (system+user+assistant): 3 [END_ID] tokens,
    answer bắt đầu tại token ngay sau cái [END_ID] thứ 3.
    """
    input_ids = example["input_ids"]

    # Phòng thủ: unwrap nếu input_ids vẫn còn là dict (BatchEncoding)
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    input_ids = [int(t) for t in input_ids]

    end_id_token = tokenizer.convert_tokens_to_ids("[END_ID]")

    # Tổng số [END_ID] trong sequence → answer bắt đầu sau [END_ID] cuối cùng
    total_end_ids = sum(1 for t in input_ids if t == end_id_token)

    query_mask = []
    occurrence = 0
    is_answer  = False

    for t in input_ids:
        if not is_answer:
            query_mask.append(0)
        else:
            query_mask.append(1)

        if t == end_id_token:
            occurrence += 1
            if occurrence == total_end_ids:   # vừa qua [END_ID] cuối → token tiếp = answer
                is_answer = True

    example["input_ids"]  = input_ids   # đảm bảo lưu lại flat list
    example["query_mask"] = query_mask
    return example



def prepare_data(args):
    context_length = args.context_length
    path_to_save = args.path_to_data_store
    cache_dir = args.huggingface_cache_dir

    tokenizer = get_tokenizer()

    print("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)["train"]
    print(f"Loaded {len(dataset)} samples.")

    # Map: filter thinking + apply chat template + tokenize
    tokenized = dataset.map(
        lambda sample: process_sample(sample, tokenizer, context_length),
        remove_columns=dataset.column_names,
        num_proc=args.num_workers,
        batched=False,
        desc="Applying chat template & tokenizing",
    )

    # Filter: bỏ sample có token length > context_length
    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda sample: len(sample["input_ids"]) <= context_length,
        num_proc=args.num_workers,
        desc=f"Filtering samples > {context_length} tokens",
    )
    print(f"Filtered {before - len(tokenized)} samples exceeding {context_length} tokens "
          f"({len(tokenized)} remaining).")

    # Map: tạo answer mask (query_mask)
    tokenized = tokenized.map(
        lambda sample: get_answer_mask(sample, tokenizer),
        num_proc=args.num_workers,
        batched=False,
        desc="Building answer mask",
    )

    # Train / test split
    split = tokenized.train_test_split(
        test_size=args.test_split_pct,
        seed=args.dataset_split_seed,
    )
    print(f"Train: {len(split['train'])} | Test: {len(split['test'])} samples")

    # Save to disk
    start = time.time()
    split.save_to_disk(path_to_save)
    print(f"Saved to: {path_to_save}  ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data(args)
