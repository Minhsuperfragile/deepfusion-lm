from sympy.physics.units import l
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForMaskedLM, get_scheduler
from datasets import load_from_disk
from accelerate import Accelerator
from tqdm import tqdm
from tokenizer import get_tokenizer

from model import DeepfusionConfig, DeepfusionLM
from dataclasses import dataclass
from dotenv import load_dotenv; load_dotenv()
from transformers import AutoTokenizer

TOKENIZED_DATA_PATH = os.getenv("PROCESSED_PRETRAIN_DATA_PATH", "")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

@dataclass
class PretrainConfig:
    per_gpu_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = 'cosine'
    num_warmup_steps: int = 1000
    num_training_steps: int = 10000

    logging_steps: int = 1
    # checkpoint_interval: int = 1000
    max_grad_norm: float = 1.0

# Start accelerator 
checkpoint_dir = "./checkpoints"
accelerator = Accelerator(
    project_dir=checkpoint_dir,
    log_with="tensorboard",
    mixed_precision="fp16" # Enable mixed precision for faster training on CUDA
)
path_to_experiment = "my_experiment"
accelerator.init_trackers(path_to_experiment)
# accelerator.log(
#     {
#         "loss": loss.item(),
#         "lr": lr
#     },
#     step=global_step
# )

modelc = DeepfusionConfig()
model = DeepfusionLM(modelc)
pretrainc = PretrainConfig()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

# Define dataloader 
mini_batchsize = pretrainc.per_gpu_batch_size // pretrainc.gradient_accumulation_steps

def collate_fn(batch):
    tokens = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
    return {"input_ids": tokens}

tokenized_data = load_from_disk(TOKENIZED_DATA_PATH)
assert isinstance(tokenized_data, Dataset) , f"tokenized_data is {type(tokenized_data)}, expect a Dataset"

train_dataloader = DataLoader(tokenized_data, 
                              batch_size=mini_batchsize,
                              collate_fn=collate_fn, 
                              shuffle=True,
                              pin_memory=True) # Speed up data transfer to GPU

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=pretrainc.learning_rate, 
                              weight_decay=pretrainc.weight_decay)

scheduler = get_scheduler(
    name=pretrainc.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=pretrainc.num_warmup_steps,
    num_training_steps=pretrainc.num_training_steps,
)

### Loss Function ###
loss_func = nn.CrossEntropyLoss(reduction="none")

### Prepare Everything ###
model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

### Start Training ###
train = True
completed_steps = 0
progress_bar = tqdm(range(completed_steps, pretrainc.num_training_steps), disable=not accelerator.is_local_main_process)

while train:

    ### Keep Track of Accumulated Mini-Steps ###
    accumulate_steps = 0
    
    ### Accumulated Loss ###
    accumulate_loss = 0
    
    for batch in train_dataloader:        

        ### Grab Input IDs ###
        input_ids = batch["input_ids"].to(accelerator.device)

        ### Create Attention Mask (True for valid tokens, False for padding) ###
        batch_size, seq_len = input_ids.shape
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id)

        ### Random sample t to mask each token with that probability ###
        t = torch.rand(batch_size, 1, device=accelerator.device).expand(batch_size, seq_len).clamp_min(1e-5)
        mask = torch.bernoulli(t).bool()

        ### IMPORTANT: Do not mask padding tokens! ###
        mask = mask & attention_mask

        ### Mask Data and Dont Compute Loss for Unmasked Data ###
        masked_input_ids = input_ids.masked_fill(mask, tokenizer.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)

        ### Compute Logits ###
        logits = model(input_ids=masked_input_ids, attention_mask=attention_mask)
  
        ### Compute Loss (per token) ###
        num_classes = logits.shape[-1]
        loss = loss_func(logits.reshape(batch_size*seq_len, num_classes),
                         labels.flatten())
        
        ### Scale loss by t. As t gets larger, we have more mask tokens and its becomes a tougher problem ###
        ### So naturally samples with large t, will have a worse loss. Just to make it fair we scale our ###
        ### loss per sample by the t ###
        loss = loss.reshape(batch_size, seq_len) / t
        loss = loss.mean()

        ### Scale Loss by Gradient Accumulation Steps ###
        loss = loss / pretrainc.gradient_accumulation_steps
        accumulate_loss += loss.detach()

        ### Compute Gradients ###
        accelerator.backward(loss)

        accumulate_steps += 1

        if accumulate_steps % pretrainc.gradient_accumulation_steps == 0:

            ### Update Model ###
            accelerator.clip_grad_norm_(model.parameters(), pretrainc.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            ### Update Scheduler ###
            scheduler.step()

            ### Log Results!! ###
            if completed_steps % pretrainc.logging_steps == 0:
                
                accumulate_loss = accumulate_loss.detach()
            
                if accelerator.state.num_processes > 1:

                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))

                log = {"train_loss": accumulate_loss,
                       "learning_rate": scheduler.get_last_lr()[0]}

                logging_string = f"[{completed_steps}/{pretrainc.num_training_steps}] Training Loss: {accumulate_loss}"
                
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                accelerator.log(log, step=completed_steps) 
            
            # ### Checkpoint Model (Only need main process for this) ###
            # if (completed_steps % pretrainc.checkpoint_interval == 0):
                
            #     ### Save Checkpoint ### 
            #     path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")

            #     if accelerator.is_main_process:
            #         progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

            #     ### Make sure that all processes have caught up before saving checkpoint! ###
            #     accelerator.wait_for_everyone()

            #     ### Save checkpoint using only the main process ###
            #     if accelerator.is_main_process:
            #         accelerator.save_state(output_dir=path_to_checkpoint)
            
            if completed_steps >= pretrainc.num_training_steps:
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
