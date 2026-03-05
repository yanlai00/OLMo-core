import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
print("Loading Dolci-Instruct-SFT dataset...")
dataset = load_dataset("allenai/Dolci-Instruct-SFT", split="train")

# Filter for math data (id starts with "personas_math_easy")
print("Filtering for personas_math_easy...")
math_dataset = dataset.filter(lambda x: x["id"].startswith("personas_math_easy"))
print(f"Filtered dataset size: {len(math_dataset):,} examples")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

# Define chat template for OLMo 3 Instruct
def format_messages(messages):
    """Format conversation messages with OLMo 3 style special tokens."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}<|endoftext|>\n"
        elif role == "system":
            text += f"<|system|>\n{content}\n"
    return text

# Tokenize all examples
print("Tokenizing...")
all_token_ids = []
for i, example in enumerate(math_dataset):
    text = format_messages(example["messages"])
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    all_token_ids.extend(token_ids)

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1:,} examples...")

print(f"Total tokens: {len(all_token_ids):,}")

# Write to numpy file
output_path = "dolci_math_easy_sft.npy"
data_mmap = np.memmap(output_path, mode="w+", dtype=np.uint32, shape=(len(all_token_ids),))
data_mmap[:] = all_token_ids
data_mmap.flush()
print(f"Saved to {output_path}")