"""
Launch with:
    # Single GPU
    python sft_lora.py

    # Multi-GPU with accelerate
    accelerate launch sft_lora.py

    # Multi-GPU with torchrun
    torchrun --nproc-per-node=4 sft_lora.py
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAD_TOKEN = "<|padding|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
IGNORE_INDEX = -100


def format_messages_to_text(messages: List[Dict], tokenizer) -> str:
    """Format conversation messages to text using OLMo 3 chat template."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}{tokenizer.eos_token}\n"
        elif role == "system":
            text += f"<|system|>\n{content}\n"
    return text


def tokenize_conversation(
    example: Dict,
    tokenizer,
    max_length: int,
) -> Dict:
    """Tokenize a conversation example for causal LM training."""
    messages = example["messages"]

    # Format messages to text
    text = format_messages_to_text(messages, tokenizer)

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def prepare_dataset(
    tokenizer,
    dataset_name: str = "allenai/Dolci-Instruct-SFT",
    dataset_filter: Optional[str] = None,
    max_length: int = 4096,
    num_proc: int = 8,
) -> Dataset:
    """Load and prepare a conversation dataset for SFT training."""
    logger.info(f"Loading dataset: {dataset_name}...")

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Dataset size: {len(dataset):,} examples")

    # Filter by id prefix if specified
    if dataset_filter:
        logger.info(f"Filtering for '{dataset_filter}' examples...")
        dataset = dataset.filter(
            lambda x: x["id"].startswith(dataset_filter),
            num_proc=num_proc,
        )
        logger.info(f"Filtered dataset size: {len(dataset):,} examples")

    # Tokenize
    logger.info("Tokenizing dataset...")
    dataset = dataset.map(
        lambda x: tokenize_conversation(x, tokenizer, max_length),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    return dataset


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OLMo-3-7B-Instruct-SFT on math data with LoRA")

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/Olmo-3-7B-Instruct-SFT",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Tokenizer name (defaults to model name)",
    )

    # Data arguments
    parser.add_argument("--dataset-name", type=str, default="allenai/Dolci-Instruct-SFT", help="HuggingFace dataset name")
    parser.add_argument("--dataset-filter", type=str, default=None, help="Filter dataset by id prefix (e.g., 'personas_math_easy')")
    parser.add_argument("--max-length", type=int, default=4096, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./math-sft-lora-hf", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Max checkpoints to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )

    # Precision arguments
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (QLoRA)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="olmo3-math-sft", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    # Other
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Initialize wandb if enabled
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger.info(f"W&B logging enabled - Project: {args.wandb_project}")
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            args.wandb = False

    # Set seed
    set_seed(args.seed)

    # Determine device
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    logger.info(f"Loading model: {args.model_name}")

    # Load tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with appropriate precision
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto" if local_rank == -1 else {"": local_rank},
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype", None)
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Prepare model for k-bit training if using quantization
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Enable gradient checkpointing
    if args.gradient_checkpointing and not (args.load_in_4bit or args.load_in_8bit):
        model.gradient_checkpointing_enable()

    # Configure LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    logger.info(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Prepare dataset
    train_dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_filter=args.dataset_filter,
        max_length=args.max_length,
    )

    logger.info(f"Training dataset size: {len(train_dataset):,}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="wandb" if args.wandb else "none",
        run_name=args.wandb_run_name if args.wandb else None,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed!")
    logger.info(f"To load the fine-tuned model:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  from transformers import AutoModelForCausalLM")
    logger.info(f"  base_model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")


if __name__ == "__main__":
    main()
