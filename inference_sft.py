import os
import sys
import logging
import argparse
import torch

### Vietnamese text needs UTF-8; legacy Windows consoles default to cp1252 ###
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
from rich.live import Live
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

from model import DeepfusionConfig, DeepfusionLM

### Dont Need any Verbose from Transformers ###
logging.getLogger("transformers").setLevel(logging.ERROR)

### Defaults pointing at this project ###
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_PATH = os.path.join(_DIR, "DeepfusionLM_tokenizer")
DEFAULT_CHECKPOINT_PATH = os.path.join(_DIR, "my_experiment", "checkpoint_sft_model")


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device="cuda"):

    ### Load Tokenizer (already configured with chat template + special tokens) ###
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = "[EOS]"  # match training setup

    ### Instantiate DeepfusionLM with vocab size from the tokenizer ###
    config = DeepfusionConfig()
    config.vocab_size = len(tokenizer)
    model = DeepfusionLM(config)

    ### Accelerate `save_state` writes a directory; resolve model.safetensors inside it ###
    if os.path.isdir(checkpoint_path):
        weights_file = os.path.join(checkpoint_path, "model.safetensors")
    else:
        weights_file = checkpoint_path

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Could not find weights at: {weights_file}")

    ### Load the SFT checkpoint ###
    state_dict = load_file(weights_file)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded SFT weights from: {weights_file}")

    model.to(device)
    model.eval()

    return model, tokenizer


def prepare_unconditional_tokens_for_inference(seq_len, mask_token_id, device="cuda"):
    input_tokens = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    # attention_mask: 1/True = valid token (see model.py forward); attend to everything, as in SFT
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    return input_tokens, mask, attention_mask


def prepare_conditional_tokens_for_inference(seq_len, tokenizer, prompt, device="cuda"):

    ### Start Inference with Chat Template ###
    chat_template = [
        {"role": "user", "content": prompt}
    ]

    ### Tokenize (chat template already inserts special tokens like [BOS]) ###
    tokenized = tokenizer.apply_chat_template(
        chat_template,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )

    ### apply_chat_template may return list[int], a BatchEncoding, or a raw
    ### tokenizers.Encoding depending on version/path — normalize to a flat id list. ###
    if isinstance(tokenized, list):
        token_ids = tokenized
    elif hasattr(tokenized, "ids"):           # tokenizers.Encoding
        token_ids = tokenized.ids
    elif hasattr(tokenized, "input_ids"):     # BatchEncoding
        token_ids = tokenized.input_ids
    else:
        token_ids = tokenized["input_ids"]    # dict-like fallback
    if len(token_ids) > 0 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]              # unwrap batch dimension if present
    token_ids = [int(t) for t in token_ids]

    ### Convert to Tensor and Move to Device ###
    prompt_tokens = torch.tensor(token_ids, dtype=torch.long).to(device)

    if len(prompt_tokens) >= seq_len:
        raise ValueError(
            f"Prompt ({len(prompt_tokens)} tokens) does not fit in seq_len={seq_len}. "
            f"Increase --seq_len."
        )

    ### Create Unconditional Token Input ###
    input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
        seq_len, tokenizer.mask_token_id, device
    )

    ### Inset our Prompt Tokens into x ###
    input_tokens[0, :len(prompt_tokens)] = prompt_tokens

    ### Set Mask to False on Prompt Tokens (Cannot be Updated During Generation) ###
    mask[0, :len(prompt_tokens)] = False

    return input_tokens, mask, attention_mask


def format_display_for_qa(user_text, assistant_text):
    output = Text()
    output.append("USER: ", style="bold green")
    output.append(user_text + "\n\n")
    output.append("ASSISTANT: ", style="bold cyan")
    output.append(assistant_text, style="white")
    return output


def format_display_for_unconditional(gen_text):
    output = Text()
    output.append("Unconditional Generation: \n\n", style="bold green")
    output.append(gen_text, style="white")
    return output


def clean_text(raw_text: str) -> str:
    return (
        raw_text.replace("user", "")
        .replace("assistant", "")
        .strip()
    )


@torch.no_grad()
def inference(model,
              tokenizer,
              input_tokens,
              mask,
              attention_mask,
              num_steps,
              remasking="random",
              temperature=0.0,
              device="cuda",
              prompt=None,
              show_mask=True):

    ### Nice Printing Stuff ##
    console = Console(highlight=False)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:

        ### What Controls our Progress Bar ###
        task = progress.add_task("Generating...", total=num_steps)

        ### Get Timesteps for Inference ###
        times = torch.linspace(1, 0, num_steps + 1, device=device)

        with Live("", refresh_per_second=5, console=console) as live:
            for t, s in zip(times[:-1], times[1:]):

                ### Compute Logits (DeepfusionLM.forward returns logits directly) ###
                logits = model(input_tokens, attention_mask=attention_mask)

                ### Predict Gen Tokens from Masked Positions ###
                ### LLaDA Alg. 4/5 use greedy argmax ("we employ greedy sampling"). ###
                ### temperature > 0 opts into stochastic sampling for experimentation. ###
                if temperature and temperature > 0.0:
                    probs = torch.softmax(logits[mask] / temperature, dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    input_tokens[mask] = logits[mask].argmax(dim=-1)

                ### All Tokens are Randomly Remasked ###
                if remasking == "random":

                    ### For Every Position, sample a value betweewn 0 and 1 ###
                    remask_probs = torch.rand_like(mask, dtype=torch.float, device=device)

                    ### If less than proportion token is selected to be remasked ###
                    remask_probs = (remask_probs < s/t)

                    ### Only replace if our mask token was previous True and is again True ###
                    ### once a token is false (no more masking) it is here to stay! ###
                    mask = mask & remask_probs

                    ### Set those tokens back to mask ###
                    input_tokens[mask] = tokenizer.mask_token_id

                ### Low confidence Tokens are Randomly Remasked ###
                elif remasking == "low_confidence":

                    ### Compute Probs for all Tokens ###
                    probs_all = torch.nn.functional.softmax(logits, dim=-1)

                    ### Confidence c^i = probability of the predicted token (argmax when greedy). ###
                    ### probs_all: 1 x seq_len x vocab_size
                    ### input_tokens: 1 x seq_len
                    chosen_token_probs = torch.gather(probs_all, dim=-1,
                                                      index=input_tokens.unsqueeze(-1)).squeeze(-1)

                    ### Make sure to set all tokens already selected to not be remasked to again ###
                    ### not be selected to be remasked. We can just set them to 1 because we want ###
                    ### low confidence (prob) tokens to be replaced! (set False to 1) ###
                    chosen_token_probs[~mask] = 1.0

                    ### Compute Proportion of Tokens to Remask out of the tokens that are currently masked ###
                    num_to_remask = int((s/t) * mask.sum().item())

                    if num_to_remask > 0:

                        ### Find the lowest prob tokens ###
                        lowest_confidence_idx = torch.topk(chosen_token_probs, num_to_remask, largest=False).indices

                        ### Create a New Mask (where everything is set to False) ###
                        new_mask = torch.zeros_like(mask)

                        ### Set the lowest confidence tokens to be remasked ###
                        new_mask[0, lowest_confidence_idx] = True
                        mask = new_mask

                        ### Update our Input Tokens with Mask Tokens ###
                        input_tokens[mask] = tokenizer.mask_token_id

                if show_mask:
                    ### Get all of the Tokens ###
                    decoded_tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])

                    ### Keep [MASK] tokens, drop all other special tokens ###
                    cleaned_tokens = []
                    for tok in decoded_tokens:
                        if tok == tokenizer.mask_token:  # keep mask tokens
                            cleaned_tokens.append(tok)
                        elif tok in tokenizer.all_special_tokens:  # drop all other specials
                            continue
                        else:
                            cleaned_tokens.append(tok)

                    ### Put all the tokens back together into a string ###
                    decoded_after = tokenizer.convert_tokens_to_string(cleaned_tokens)

                else:
                    decoded_after = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]

                if prompt is None:
                    format_text = format_display_for_unconditional(decoded_after)
                else:
                    ### Remove Prompt Text from Assistant Text ###
                    assistant_text = decoded_after.replace(prompt, "").strip()
                    ### Remove Keywords user and assistant ###
                    assistant_text = clean_text(assistant_text)
                    format_text = format_display_for_qa(prompt, assistant_text)
                live.update(format_text)
                progress.update(task, advance=1)

    ### Print final result after the transient live view closes ###
    console.print(format_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference DeepfusionLM (post-SFT)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Accelerate save_state dir (containing model.safetensors) or a .safetensors file")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--strategy", type=str, default="low_confidence", choices=["random", "low_confidence"],
                        help="Remasking strategy: low_confidence (LLaDA Alg. 5, recommended) or random (Alg. 4)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy argmax (LLaDA default). >0 samples from softmax(logits/temperature).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    ### Load Model ###
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path,
                                                args.tokenizer_path,
                                                args.device)

    if args.prompt is None:
        ### Prepare Unconditional Inference Inputs ###
        input_tokens, mask, attention_mask = prepare_unconditional_tokens_for_inference(
            args.seq_len,
            mask_token_id=tokenizer.mask_token_id,
            device=args.device)
    else:
        ### Prepare Conditional Inference Inputs ###
        input_tokens, mask, attention_mask = prepare_conditional_tokens_for_inference(
            args.seq_len,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device)

    inference(model,
              tokenizer,
              input_tokens,
              mask,
              attention_mask,
              args.num_steps,
              remasking=args.strategy,
              temperature=args.temperature,
              device=args.device,
              prompt=args.prompt)
