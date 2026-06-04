import os
import gc
import argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, get_scheduler
from safetensors.torch import load_file
from accelerate import Accelerator
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from model import DeepfusionConfig, DeepfusionLM

@dataclass
class ContinuedPretrainConfig:
    # Dataset and paths
    dataset_path: str = os.getenv("CONTINUED_PRETRAIN_RAW_DATA_PATH", "/mnt/data1/tm/deepfusion-lm/dataset/vietnamese-history-qa")
    tokenized_dataset_path: str = os.getenv("CONTINUED_PRETRAIN_TOKENIZED_DATA_PATH", "/mnt/data1/tm/deepfusion-lm/dataset/vietnamese-history-qa-tokenized")
    model_weight_dir: str = os.getenv("CONTINUED_PRETRAIN_MODEL_WEIGHT_DIR", "my_experiment/checkpoint_120000")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "DeepfusionLM_tokenizer")
    experiment_name: str = os.getenv("CONTINUED_PRETRAIN_EXPERIMENT_NAME", "continued_pretrain")
    working_directory: str = os.getenv("CONTINUED_PRETRAIN_WORKING_DIR", "my_experiment")

    # Training hyperparameters
    per_gpu_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    num_training_steps: int = 6250
    max_grad_norm: float = 1.0

    # Intervals and logging
    logging_steps: int = 10
    evaluation_interval: int = 500
    checkpoint_interval: int = -1  # Set to > 0 to save intermediate checkpoints (e.g. 1000)

    # Dataset processing
    test_split_pct: float = 0.01
    num_proc: int = 8
    seed: int = 42


def get_config() -> ContinuedPretrainConfig:
    config = ContinuedPretrainConfig()

    parser = argparse.ArgumentParser(description="Continued Pretraining on Domain Dataset")

    # Paths
    parser.add_argument("--dataset_path", type=str, default=config.dataset_path)
    parser.add_argument("--tokenized_dataset_path", type=str, default=config.tokenized_dataset_path,
                        help="Where to save/load the pre-tokenized dataset")
    parser.add_argument("--model_weight_dir", type=str, default=config.model_weight_dir)
    parser.add_argument("--tokenizer_path", type=str, default=config.tokenizer_path)
    parser.add_argument("--experiment_name", type=str, default=config.experiment_name)
    parser.add_argument("--working_directory", type=str, default=config.working_directory)

    # Training hyperparams
    parser.add_argument("--per_gpu_batch_size", type=int, default=config.per_gpu_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--lr_scheduler_type", type=str, default=config.lr_scheduler_type)
    parser.add_argument("--num_warmup_steps", type=int, default=config.num_warmup_steps)
    parser.add_argument("--num_training_steps", type=int, default=config.num_training_steps)
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm)

    # Intervals
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps)
    parser.add_argument("--evaluation_interval", type=int, default=config.evaluation_interval)
    parser.add_argument("--checkpoint_interval", type=int, default=config.checkpoint_interval)

    # Dataset details
    parser.add_argument("--test_split_pct", type=float, default=config.test_split_pct)
    parser.add_argument("--num_proc", type=int, default=config.num_proc)
    parser.add_argument("--seed", type=int, default=config.seed)

    args = parser.parse_args()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def prepare_tokenized_dataset(dataset_path, tokenized_dataset_path, tokenizer, context_length, num_proc=8):
    if os.path.exists(tokenized_dataset_path):
        print(f"Loading pre-tokenized dataset from {tokenized_dataset_path}")
        return load_from_disk(tokenized_dataset_path)

    print(f"Tokenizing dataset from {dataset_path}...")
    raw = load_from_disk(dataset_path)

    if hasattr(raw, "keys"):
        raw = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]

    def tokenize_and_chunk(examples):
        all_ids = []
        for messages in examples["messages"]:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
            all_ids.extend(ids)

        result_chunks = []
        for i in range(0, len(all_ids), context_length):
            chunk = all_ids[i : i + context_length]
            if len(chunk) == context_length:
                result_chunks.append(chunk)

        return {"input_ids": result_chunks}

    tokenized = raw.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=raw.column_names,
    )

    print(f"Saving tokenized dataset to {tokenized_dataset_path}...")
    tokenized.save_to_disk(tokenized_dataset_path)
    print(f"Done. Created {len(tokenized)} packed sequences of length {context_length}.")
    return tokenized


def load_model_weights(model, weight_dir):
    safetensors_path = os.path.join(weight_dir, "model.safetensors")
    pytorch_path = os.path.join(weight_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {safetensors_path}")
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {pytorch_path}")
    elif weight_dir.endswith(".safetensors") and os.path.exists(weight_dir):
        state_dict = load_file(weight_dir)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from file {weight_dir}")
    elif weight_dir.endswith(".bin") and os.path.exists(weight_dir):
        state_dict = torch.load(weight_dir, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from file {weight_dir}")
    else:
        raise FileNotFoundError(f"Could not load weights from weight_dir: {weight_dir}")


def evaluate(model, dataloader, accelerator, tokenizer, loss_func):
    model.eval()
    val_loss = 0.0
    num_batches = 0

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id

    for batch in dataloader:
        input_ids = batch["input_ids"].to(accelerator.device)
        batch_size, seq_len = input_ids.shape
        attention_mask = (input_ids != pad_id)

        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
        mask = torch.bernoulli(t).bool()
        mask = mask & attention_mask

        masked_input_ids = input_ids.masked_fill(mask, mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        with torch.inference_mode():
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)

        num_classes = logits.shape[-1]
        loss = loss_func(logits.reshape(batch_size * seq_len, num_classes), labels.flatten())
        loss = loss.reshape(batch_size, seq_len) / t
        loss = loss.mean().detach()

        if accelerator.num_processes > 1:
            loss = torch.mean(accelerator.gather_for_metrics(loss))

        val_loss += loss.item()
        num_batches += 1

    model.train()
    return val_loss / max(num_batches, 1)


def main():
    config = get_config()

    path_to_experiment = os.path.join(config.working_directory, config.experiment_name)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        log_with="tensorboard",
        mixed_precision="fp16"
    )

    accelerator.init_trackers(config.experiment_name)

    # Load tokenizer (already fully configured with chat template and special tokens)
    accelerator.print(f"Loading tokenizer from: {config.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # Load model
    model_config = DeepfusionConfig()
    model_config.vocab_size = len(tokenizer)
    model = DeepfusionLM(model_config)
    load_model_weights(model, config.model_weight_dir)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Trainable Parameters:", params)

    # Tokenize dataset (skips if already done)
    tokenized_dataset = prepare_tokenized_dataset(
        dataset_path=config.dataset_path,
        tokenized_dataset_path=config.tokenized_dataset_path,
        tokenizer=tokenizer,
        context_length=model_config.max_seq_len,
        num_proc=config.num_proc,
    )

    accelerator.print(f"Total packed sequences: {len(tokenized_dataset)}")

    # Train/test split
    split_data = tokenized_dataset.train_test_split(test_size=config.test_split_pct, seed=config.seed)
    train_dataset = split_data["train"]
    eval_dataset = split_data["test"]

    def collate_fn(batch):
        tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        return {"input_ids": tokens}

    mini_batchsize = max(1, config.per_gpu_batch_size // config.gradient_accumulation_steps)

    eval_dataloader = DataLoader(eval_dataset, batch_size=mini_batchsize, collate_fn=collate_fn, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps,
    )
    loss_func = nn.CrossEntropyLoss(reduction="none")

    model, optimizer, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, eval_dataloader, scheduler
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id

    completed_steps = 0
    progress_bar = tqdm(
        range(completed_steps, config.num_training_steps),
        disable=not accelerator.is_local_main_process,
    )

    train_active = True
    epoch = 0

    while train_active:
        epoch += 1
        accumulate_steps = 0
        accumulate_loss = 0.0

        train_dataloader = DataLoader(
            train_dataset, batch_size=mini_batchsize, collate_fn=collate_fn, shuffle=True
        )
        train_dataloader = accelerator.prepare_data_loader(train_dataloader)

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(accelerator.device)
            batch_size, seq_len = input_ids.shape
            attention_mask = (input_ids != pad_id)

            t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool()
            mask = mask & attention_mask

            masked_input_ids = input_ids.masked_fill(mask, mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)

            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)

            num_classes = logits.shape[-1]
            loss = loss_func(logits.reshape(batch_size * seq_len, num_classes), labels.flatten())
            loss = loss.reshape(batch_size, seq_len) / t
            loss = loss.mean()

            loss = loss / config.gradient_accumulation_steps
            accumulate_loss += loss.detach()
            accelerator.backward(loss)
            accumulate_steps += 1

            if accumulate_steps % config.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                completed_steps += 1
                progress_bar.update(1)

                if completed_steps % config.logging_steps == 0:
                    loss_val = accumulate_loss.detach()
                    if accelerator.state.num_processes > 1:
                        loss_val = torch.mean(accelerator.gather_for_metrics(loss_val))

                    logging_string = f"[{completed_steps}/{config.num_training_steps}] Train Loss: {loss_val.item():.4f}"
                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)

                    accelerator.log(
                        {"train_loss": loss_val.item(), "learning_rate": scheduler.get_last_lr()[0]},
                        step=completed_steps,
                    )

                if completed_steps % config.evaluation_interval == 0:
                    accelerator.print("Evaluating model...")
                    val_loss = evaluate(model, eval_dataloader, accelerator, tokenizer, loss_func)
                    val_string = f"[{completed_steps}/{config.num_training_steps}] Val Loss: {val_loss:.4f}"
                    if accelerator.is_main_process:
                        progress_bar.write(val_string)
                    accelerator.log({"val_loss": val_loss}, step=completed_steps)

                if config.checkpoint_interval > 0 and completed_steps % config.checkpoint_interval == 0:
                    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
                    if accelerator.is_main_process:
                        progress_bar.write(f"Saving checkpoint to {path_to_checkpoint}")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output_dir=path_to_checkpoint)

                accumulate_loss = 0.0

                if completed_steps >= config.num_training_steps:
                    train_active = False
                    if accelerator.is_main_process:
                        progress_bar.write("Completed Training!!")
                    break

        del train_dataloader
        gc.collect()

    final_checkpoint = os.path.join(path_to_experiment, "final_model")
    if accelerator.is_main_process:
        print(f"Saving final model to {final_checkpoint}")
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=final_checkpoint)
    accelerator.end_training()


if __name__ == "__main__":
    main()
