import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm

from model import DeepfusionConfig, DeepfusionLM

# ─── Paths ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_PATH  = os.path.join(_DIR, "DeepfusionLM_tokenizer")
DEFAULT_CHECKPOINT_PATH = os.path.join(_DIR, "checkpoint-pretrained-with-history-qa", "model.safetensors")
DEFAULT_DATA_PATH       = os.path.join(_DIR, "data", "tokenized-sft")
DEFAULT_OUTPUT_DIR      = os.path.join(_DIR, "sft_output")


def SFTCollator(tokenizer):
    eos_token = tokenizer.eos_token_id

    def _collate_fn(batch):
        inputs = [torch.tensor(b["input_ids"]) for b in batch]
        query_masks = [torch.tensor(b["query_mask"]) for b in batch]

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value=eos_token, batch_first=True)
        query_masks = torch.nn.utils.rnn.pad_sequence(query_masks, padding_value=1, batch_first=True)
        
        return {"input_ids": inputs, "query_mask": query_masks}
        
    return _collate_fn

def parse_args():
    ### PARSE COMMAND LINE ARGS ###
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--experiment_name", 
        required=True, 
        type=str
    )

    parser.add_argument(
        "--working_directory", 
        required=True, 
        type=str
    )
    
    parser.add_argument(
        "--path_to_pretrained_checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        type=str
    )

    parser.add_argument(
        "--tokenizer_path",
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to local DeepfusionLM tokenizer directory",
        type=str
    )

    #########################
    ### DATASET ARGUMENTS ###
    #########################

    parser.add_argument(
        "--path_to_prepped_data",
        default=DEFAULT_DATA_PATH,
        help="Path to data prepared in `prepare_sft_data.py`",
        type=str
    )

    parser.add_argument(
        "--num_workers",
        help="Number of workers for dataloading",
        default=4, 
        type=int
    )

    ##############################
    ### TRAINING CONFIGURATION ###
    ##############################

    parser.add_argument(
        "--per_gpu_batch_size",
        help="Overall batch size per gpu during training",
        default=16, 
        type=int
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="Splits per_gpu_batch_size by gradient_accumulation_steps",
        default=1, 
        type=int
    )

    parser.add_argument(
        "--num_training_steps", 
        help="Number of optimizer steps to train (bị ghi đè nếu dùng --num_epochs)",
        default=None,
        type=int
    )

    parser.add_argument(
        "--num_epochs",
        help="Số epochs để train. Nếu set thì tự tính num_training_steps từ dataset size.",
        default=None,
        type=float
    )

    parser.add_argument(
        "--max_grad_norm",
        help="Max gradient norm used for stabilizing training with gradient clipping",
        default=1.0, 
        type=float
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_steps", 
        help="Number of iterations for every log of metrics",
        default=1,
        type=int
    )

    parser.add_argument(
        "--evaluation_interval", 
        help="Number of iterations for every evaluation",
        default=2500, 
        type=int
    )

    parser.add_argument(
        "--checkpoint_interval",
        help="Number of iterations for checkpointing",
        default=10000,
        type=int
    )

    parser.add_argument(
        "--learning_rate", 
        help="Max learning rate for all Learning Rate Schedulers", 
        default=1e-5, 
        type=float
    )

    parser.add_argument(
        "--weight_decay",
        help="Weight decay constant for AdamW optimizer", 
        default=0.05, 
        type=float
    )

    #############################
    ### LOGGING CONFIGURATION ###
    #############################

    parser.add_argument(
        "--log_wandb",
        help="Enable logging to Weights & Biases",
        default=False,
        action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--wandb_project",
        help="W&B project name (chỉ dùng khi --log_wandb)",
        default="deepfusion-sft",
        type=str
    )

    args = parser.parse_args()

    return args

args = parse_args()

### Instantiate Accelerate ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(
    project_dir=path_to_experiment,
    log_with="wandb" if args.log_wandb else "tensorboard",
)
if args.log_wandb:
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.experiment_name}},
    )
else:
    accelerator.init_trackers(args.experiment_name)

### Define Tokenizer ###
tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
tokenizer.pad_token = "[EOS]"   # đảm bảo pad_token set đúng

### Load Model ###
config = DeepfusionConfig()
config.vocab_size = len(tokenizer)
model = DeepfusionLM(config)

state_dict = load_file(args.path_to_pretrained_checkpoint)
model.load_state_dict(state_dict, strict=True)
accelerator.print(f"Loaded pretrained weights from: {args.path_to_pretrained_checkpoint}")

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

## Define DataLoader ###
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps


tokenized_data = load_from_disk(args.path_to_prepped_data)
train_dataloader = DataLoader(tokenized_data["train"], 
                              batch_size=mini_batchsize,
                              collate_fn=SFTCollator(tokenizer), 
                              shuffle=True)

eval_dataloader = DataLoader(tokenized_data["test"], 
                             batch_size=mini_batchsize,
                             collate_fn=SFTCollator(tokenizer), 
                             shuffle=False)

### Define Optimizer ###
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=args.learning_rate, 
                              weight_decay=args.weight_decay)

### Compute num_training_steps (epoch mode takes priority) ###
import math
train_size       = len(tokenized_data["train"])
steps_per_epoch  = math.ceil(train_size / mini_batchsize / args.gradient_accumulation_steps)

if args.num_epochs is not None:
    args.num_training_steps = math.ceil(args.num_epochs * steps_per_epoch)
    accelerator.print(
        f"Epoch mode: {args.num_epochs} epoch(s) × {steps_per_epoch} steps/epoch "
        f"= {args.num_training_steps} total optimizer steps"
    )
elif args.num_training_steps is None:
    # fallback default = 1 epoch
    args.num_training_steps = steps_per_epoch
    accelerator.print(f"Defaulting to 1 epoch = {args.num_training_steps} steps")
else:
    epochs_equiv = args.num_training_steps / steps_per_epoch
    accelerator.print(
        f"Step mode: {args.num_training_steps} steps "
        f"(≈ {epochs_equiv:.2f} epochs, {steps_per_epoch} steps/epoch)"
    )

### Define Scheduler ###
scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)

### Loss Function ###
loss_func = nn.CrossEntropyLoss(reduction="none")

### Prepare Everything ###
model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

### Start Training ###
train = True
completed_steps = 0
progress_bar = tqdm(range(completed_steps, args.num_training_steps), disable=not accelerator.is_local_main_process)

### CSV Metrics Logger (chỉ main process ghi) ###
os.makedirs(path_to_experiment, exist_ok=True)
csv_path = os.path.join(path_to_experiment, "metrics.csv")
csv_file, csv_writer = None, None
if accelerator.is_main_process:
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "train_loss", "val_loss", "learning_rate"])
    csv_writer.writeheader()
    csv_file.flush()

while train:

    ### Keep Track of Accumulated Mini-Steps ###
    accumulate_steps = 0
    
    ### Accumulated Loss ###
    accumulate_loss = 0
    
    for batch in train_dataloader:        

        ### Grab Input IDs ###
        input_ids = batch["input_ids"].to(accelerator.device)
        query_mask = batch["query_mask"].to(accelerator.device)

        ### Attend to All Tokens (EVEN EOS) ###
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

        ### Random sample t to mask each token with that probability ###
        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(0.01)
        mask = torch.bernoulli(t)

        ### Mask only valid where it is not query ###
        mask = mask * query_mask
        mask = mask.bool()

        ### Mask Data and Dont Compute Loss for Unmasked Data ###
        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        ### Compute Logits ###
        # DeepfusionLM.forward() trả về logits trực tiếp (không phải dict)
        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)
  
        ### Compute Loss (per token) ###
        num_classes = logits.shape[-1]
        loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                         labels.flatten())
        
        ### Scale loss by t ###
        loss = loss.reshape(batch_size, seq_len) / t

        ### Different answers are of different lengths, so lets scale by that too ###
        answer_lengths = query_mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        answer_lengths = answer_lengths.clamp_min(1)  
        loss = loss / answer_lengths

        ### Add up all the per-token losses and average across batch ###
        loss = loss.sum(dim=1).mean()

        ### Scale Loss by Gradient Accumulation Steps ###
        loss = loss / args.gradient_accumulation_steps
        accumulate_loss += loss

        ### Compute Gradients ###
        accelerator.backward(loss)

        accumulate_steps += 1

        if accumulate_steps % args.gradient_accumulation_steps == 0:

            ### Update Model ###
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ### Update Scheduler ###
            scheduler.step()

            ### Log Results!! ###
            if completed_steps % args.logging_steps == 0:
                
                accumulate_loss = accumulate_loss.detach()
            
                if accelerator.state.num_processes > 1:

                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))

                log = {"train_loss": accumulate_loss,
                       "learning_rate": scheduler.get_last_lr()[0]}

                logging_string = f"[{completed_steps}/{args.num_training_steps}] Training Loss: {accumulate_loss}"
                
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                    # Lưu vào CSV
                    csv_writer.writerow({
                        "step":          completed_steps,
                        "train_loss":    float(accumulate_loss),
                        "val_loss":      "",
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
                    csv_file.flush()
                
                accelerator.log(log, step=completed_steps)

            ### Evaluation Loop ###
            if completed_steps % args.evaluation_interval == 0:
                if accelerator.is_main_process:
                    progress_bar.write("Evaluating Model!!")
                
                model.eval()

                ### Dictionary to Store Results ###
                log = {"val_loss": 0}

                ### Iterate Data ###
                num_losses = 0
                for batch in tqdm(eval_dataloader, disable=not accelerator.is_main_process):
                    
                    input_ids = batch["input_ids"].to(accelerator.device)
                    query_mask = batch["query_mask"].to(accelerator.device)

                    batch_size, seq_len = input_ids.shape
                    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=accelerator.device)

                    batch_size, seq_len = input_ids.shape
                    t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len)
                    mask = torch.bernoulli(t).bool()

                    mask = mask * query_mask
                    mask = mask.bool()

                    masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
                    labels = input_ids.masked_fill(~mask, -100)

                    with torch.inference_mode():
                        # DeepfusionLM.forward() trả về logits trực tiếp
                        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                    
                    num_classes = logits.shape[-1]
                    loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                                     labels.flatten())
                    
                    loss = loss.reshape(batch_size, seq_len) / t
                    answer_lengths = query_mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
                    answer_lengths = answer_lengths.clamp_min(1)  
                    loss = loss / answer_lengths
                    loss = loss.sum(dim=1).mean()

                    loss = loss.detach()
                    if accelerator.num_processes > 1:
                        loss = torch.mean(accelerator.gather_for_metrics(loss))

                    log["val_loss"] += loss
                    num_losses += 1
                
                ### Divide Log by Num Losses ###
                log["val_loss"] = log["val_loss"] / num_losses

                ## Print to Console ###
                logging_string = f"[{completed_steps}/{args.num_training_steps}] Validation Loss: {log['val_loss']}"
        
                ### Print out Log ###
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                    # Lưu val_loss vào CSV (row riêng)
                    csv_writer.writerow({
                        "step":          completed_steps,
                        "train_loss":    "",
                        "val_loss":      float(log["val_loss"]),
                        "learning_rate": scheduler.get_last_lr()[0],
                    })
                    csv_file.flush()
                
                accelerator.log(log, step=completed_steps)

                model.train()
            
            ### Checkpoint Model (Only need main process for this) ###
            if (completed_steps % args.checkpoint_interval == 0):
                
                ### Save Checkpoint ### 
                path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

                if accelerator.is_main_process:
                    progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

                ### Make sure that all processes have caught up before saving checkpoint! ###
                accelerator.wait_for_everyone()

                ### Save checkpoint using only the main process ###
                if accelerator.is_main_process:
                    accelerator.save_state(output_dir=path_to_checkpoint)
            
            if completed_steps >= args.num_training_steps:
                train = False
                if accelerator.is_main_process:
                    progress_bar.write("Completed Training!!")
                break

            ### Iterate Progress Bar and Completed Steps ###
            completed_steps += 1
            progress_bar.update(1)

            ### Reset Loss Accumulate For Next Accumulation ###
            accumulate_loss = 0

path_to_checkpoint = os.path.join(path_to_experiment, f"final_model")
accelerator.save_state(output_dir=path_to_checkpoint)
accelerator.end_training()

if accelerator.is_main_process and csv_file is not None:
    csv_file.close()
    accelerator.print(f"✓ Metrics saved → {csv_path}")