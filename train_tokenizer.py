from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import DeepfusionConfig
import os

from dotenv import load_dotenv
load_dotenv()

CONFIG=DeepfusionConfig()

# 1. Initialize the Tokenizer with an empty BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Add a Pre-tokenizer
# This splits the text into words before the BPE algorithm applies. 
# Whitespace is standard, but you can also split on punctuation.
tokenizer.pre_tokenizer = Whitespace()

# 3. Configure the Trainer
trainer = BpeTrainer(
    vocab_size=CONFIG.vocab_size,
    special_tokens=["[UNK]", "[START_ID]", "[END_ID]", "[EOT]", "[MASK]", "[BOS]", "[EOS]"] #EOS as PAD
)

# 4. Define a Generator for Large-Scale Data Ingestion
# This streams the text line-by-line, keeping memory usage minimal.
def get_training_corpus(path):
    # Determine if it's a directory or a single file
    if os.path.isdir(path):
        # Sort to ensure deterministic order if needed
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")])
    else:
        files = [path]

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            # Example line in corpus: "L'algorithme dont nous parlons, ce qui optimise le flux, est rapide."
            for line in f:
                if line.strip(): # Skip empty lines
                    yield line

# 5. Train the Tokenizer
corpus_path = os.getenv("CORPUS_PATH", "")
print("Training started...")
tokenizer.train_from_iterator(get_training_corpus(corpus_path), trainer=trainer)
print("Training complete!")

# 6. Save the Model
tokenizer.save("DeepfusionLM_tokenizer.json")