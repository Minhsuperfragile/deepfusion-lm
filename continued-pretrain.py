import os
import argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, get_scheduler
from safetensors.torch import load_file
from accelerate import Accelerator
from tqdm import tqdm

from model import DeepfusionConfig, DeepfusionLM

@dataclass
class ContinuedPretrainConfig:
    # Dataset and paths
    dataset_path: str = "/mnt/data1/tm/deepfusion-lm/dataset/vietnamese-history-qa"
    model_weight_dir: str = "my_experiment/final_model"
    tokenizer_path: str = "DeepfusionLM_tokenizer.json"
    experiment_name: str = "continued_pretrain"
    working_directory: str = "my_experiment"
    
    # Training hyperparameters
    per_gpu_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    num_training_steps: int = 5000
    max_grad_norm: float = 1.0
    
    # Intervals and logging
    logging_steps: int = 10
    evaluation_interval: int = 500
    checkpoint_interval: int = -1  # Set to > 0 to save intermediate checkpoints (e.g. 1000)
    
    # Dataset processing
    test_split_pct: float = 0.01
    seed: int = 42


def get_config() -> ContinuedPretrainConfig:
    config = ContinuedPretrainConfig()
    
    parser = argparse.ArgumentParser(description="Continued Pretraining on History QA Dataset")
    
    # Paths
    parser.add_argument("--dataset_path", type=str, default=config.dataset_path,
                        help="Path to the tokenized dataset directory")
    parser.add_argument("--model_weight_dir", type=str, default=config.model_weight_dir,
                        help="Path to the directory containing model weights (safetensors)")
    parser.add_argument("--tokenizer_path", type=str, default=config.tokenizer_path,
                        help="Path to tokenizer JSON or directory")
    parser.add_argument("--experiment_name", type=str, default=config.experiment_name,
                        help="Experiment name for logging and checkpoints")
    parser.add_argument("--working_directory", type=str, default=config.working_directory,
                        help="Working directory to save all experiments")
                        
    # Training hyperparams
    parser.add_argument("--per_gpu_batch_size", type=int, default=config.per_gpu_batch_size,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay,
                        help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default=config.lr_scheduler_type,
                        help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=config.num_warmup_steps,
                        help="Number of warmup steps")
    parser.add_argument("--num_training_steps", type=int, default=config.num_training_steps,
                        help="Number of training steps")
    parser.add_argument("--max_grad_norm", type=float, default=config.max_grad_norm,
                        help="Max gradient norm")
                        
    # Intervals
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps,
                        help="Logging interval in steps")
    parser.add_argument("--evaluation_interval", type=int, default=config.evaluation_interval,
                        help="Evaluation interval in steps")
    parser.add_argument("--checkpoint_interval", type=int, default=config.checkpoint_interval,
                        help="Checkpoint interval in steps")
                        
    # Dataset details
    parser.add_argument("--test_split_pct", type=float, default=config.test_split_pct,
                        help="Percentage of dataset to use for validation")
    parser.add_argument("--seed", type=int, default=config.seed,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Override defaults with CLI args
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return config


def load_tokenizer(tokenizer_path, max_seq_len):
    # Check if the path is a directory or file
    if os.path.isdir(tokenizer_path):
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    else:
        tokenizer_file = tokenizer_path

    if not os.path.exists(tokenizer_file):
        # Try a few fallback paths
        fallback_paths = ["DeepfusionLM_tokenizer.json", "DeepfusionLM_tokenizer/tokenizer.json"]
        for path in fallback_paths:
            if os.path.exists(path):
                tokenizer_file = path
                break
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path} or fallbacks.")

    print(f"Loading tokenizer from: {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        model_max_length=max_seq_len
    )
    
    # Configure special tokens wrapper
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
    
    # Configure Chat Template
    start_token = "[START_ID]"
    end_token = "[END_ID]"
    eot_token = "[EOT]"
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


def load_model_weights(model, weight_dir):
    # Support direct safetensors file or a directory containing model.safetensors
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
        # Check parent directory fallback if it is my-experiment/final-model
        normalized_weight_dir = weight_dir.replace("-", "_")
        safetensors_path_norm = os.path.join(normalized_weight_dir, "model.safetensors")
        if os.path.exists(safetensors_path_norm):
            state_dict = load_file(safetensors_path_norm)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from {safetensors_path_norm}")
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
        
        # Sample mask probability t
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
        loss = loss.mean()
        
        loss = loss.detach()
        if accelerator.num_processes > 1:
            loss = torch.mean(accelerator.gather_for_metrics(loss))
            
        val_loss += loss.item()
        num_batches += 1
        
    model.train()
    return val_loss / max(num_batches, 1)


def main():
    config = get_config()
    
    # Setup Accelerator
    path_to_experiment = os.path.join(config.working_directory, config.experiment_name)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        log_with="tensorboard",
        mixed_precision="fp16"
    )
    
    # Initialize logger
    accelerator.init_trackers(config.experiment_name)
    
    # Load Model Configuration & Instantiate Model
    model_config = DeepfusionConfig()
    
    # Load Tokenizer first to adjust vocab_size if needed
    tokenizer = load_tokenizer(config.tokenizer_path, model_config.max_seq_len)
    model_config.vocab_size = len(tokenizer)
    
    model = DeepfusionLM(model_config)
    
    # Load Pretrained Weights
    load_model_weights(model, config.model_weight_dir)
    
    # Log parameter count
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print("Number of Trainable Parameters:", params)
    
    # Load and process dataset
    accelerator.print(f"Loading dataset from: {config.dataset_path}")
    dataset = load_from_disk(config.dataset_path)
    
    # Normalize DatasetDict to single Dataset
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_key = list(dataset.keys())[0]
            dataset = dataset[first_key]
            
    accelerator.print(f"Loaded raw dataset with {len(dataset)} examples.")
    
    # Tokenization and chunk packing function
    context_length = model_config.max_seq_len
    
    def tokenize_and_chunk(examples):
        all_ids = []
        for messages in examples["messages"]:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
            all_ids.extend(ids)
            
        total_len = len(all_ids)
        result_chunks = []
        for i in range(0, total_len, context_length):
            chunk = all_ids[i : i + context_length]
            if len(chunk) == context_length:
                result_chunks.append(chunk)
                
        return {"input_ids": result_chunks}
    
    # Process dataset
    accelerator.print("Tokenizing and packing dataset chunks...")
    num_proc = min(os.cpu_count() or 1, 8)
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )
    
    accelerator.print(f"Created {len(tokenized_dataset)} packed sequence chunks of length {context_length}.")
    
    # Train-test split
    split_data = tokenized_dataset.train_test_split(test_size=config.test_split_pct, seed=config.seed)
    train_dataset = split_data["train"]
    eval_dataset = split_data["test"]
    
    # Collate function
    def collate_fn(batch):
        tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        return {"input_ids": tokens}
        
    mini_batchsize = max(1, config.per_gpu_batch_size // config.gradient_accumulation_steps)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=mini_batchsize,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=mini_batchsize,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True
    )
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps
    )
    
    loss_func = nn.CrossEntropyLoss(reduction="none")
    
    # Prepare with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    # Training loop state
    completed_steps = 0
    progress_bar = tqdm(
        range(completed_steps, config.num_training_steps),
        disable=not accelerator.is_local_main_process
    )
    
    train_active = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id
    
    while train_active:
        accumulate_steps = 0
        accumulate_loss = 0.0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(accelerator.device)
            batch_size, seq_len = input_ids.shape
            attention_mask = (input_ids != pad_id)
            
            # Mask generation logic
            t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
            mask = torch.bernoulli(t).bool()
            mask = mask & attention_mask
            
            masked_input_ids = input_ids.masked_fill(mask, mask_token_id)
            labels = input_ids.masked_fill(~mask, -100)
            
            # Forward pass
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            
            # Calculate loss (scaled by mask probability t)
            num_classes = logits.shape[-1]
            loss = loss_func(logits.reshape(batch_size * seq_len, num_classes), labels.flatten())
            loss = loss.reshape(batch_size, seq_len) / t
            loss = loss.mean()
            
            # Gradient Accumulation
            loss = loss / config.gradient_accumulation_steps
            accumulate_loss += loss.detach()
            
            # Backward pass
            accelerator.backward(loss)
            accumulate_steps += 1
            
            if accumulate_steps % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer and Scheduler steps
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                completed_steps += 1
                progress_bar.update(1)
                
                # Metric Logging
                if completed_steps % config.logging_steps == 0:
                    accumulate_loss = accumulate_loss.detach()
                    if accelerator.state.num_processes > 1:
                        accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                        
                    log_data = {
                        "train_loss": accumulate_loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                    }
                    
                    logging_string = f"[{completed_steps}/{config.num_training_steps}] Training Loss: {accumulate_loss.item():.4f}"
                    if accelerator.is_main_process:
                        progress_bar.write(logging_string)
                        
                    accelerator.log(log_data, step=completed_steps)
                
                # Evaluation Loop
                if completed_steps % config.evaluation_interval == 0:
                    accelerator.print("Evaluating model...")
                    val_loss = evaluate(model, eval_dataloader, accelerator, tokenizer, loss_func)
                    
                    val_logging_string = f"[{completed_steps}/{config.num_training_steps}] Validation Loss: {val_loss:.4f}"
                    if accelerator.is_main_process:
                        progress_bar.write(val_logging_string)
                        
                    accelerator.log({"val_loss": val_loss}, step=completed_steps)
                
                # Checkpoint saving
                if config.checkpoint_interval > 0 and completed_steps % config.checkpoint_interval == 0:
                    path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
                    if accelerator.is_main_process:
                        progress_bar.write(f"Saving training checkpoint state to {path_to_checkpoint}")
                        
                    accelerator.wait_for_everyone()
                    accelerator.save_state(output_dir=path_to_checkpoint)
                
                # Reset accumulated loss
                accumulate_loss = 0.0
                
                if completed_steps >= config.num_training_steps:
                    train_active = False
                    if accelerator.is_main_process:
                        progress_bar.write("Completed Training!!")
                    break

    # Save final model state
    final_checkpoint = os.path.join(path_to_experiment, "final_model")
    if accelerator.is_main_process:
        print(f"Saving final model checkpoint state to {final_checkpoint}")
        
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=final_checkpoint)
    accelerator.end_training()


if __name__ == "__main__":
    main()
