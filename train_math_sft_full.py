"""
Continue SFT training on math data for OLMo-3-7B-Instruct-SFT.

Launch with:
    torchrun --nproc-per-node=8 train_math_sft.py [OVERRIDES...]
"""

import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 12536
    load_path: Optional[str] = None
    load_trainer_state: bool = False


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    # Build components
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config
    config_dict = config.as_config_dict()
    from olmo_core.train.callbacks import ConfigSaverCallback
    if "config_saver" in trainer.callbacks:
        trainer.callbacks["config_saver"].config = config_dict

    # Load pretrained checkpoint
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(f"Loading checkpoint from {config.load_path}...")
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

    # Train
    trainer.fit()


def build_config(args) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.dolma2()

    # OLMo 3 7B model config
    model_config = TransformerConfig.olmo3_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    # Dataset config - using packed FSL for efficient SFT
    dataset_config = NumpyPackedFSLDatasetConfig(
        paths=[args.data_path],
        sequence_length=args.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=args.work_dir,
    )

    # Data loader config
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=args.global_batch_size,
        seed=0,
        num_workers=4,
    )

    # Train module config with SFT-appropriate settings
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=args.microbatch_size,
        max_sequence_length=args.sequence_length,
        optim=AdamWConfig(
            lr=args.learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=args.warmup_steps),
    )

    # Trainer config
    trainer_config = (
        TrainerConfig(
            save_folder=args.save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=args.save_interval,
                ephemeral_save_interval=500,
                save_async=False,  # Disable async saving to reduce memory usage
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=args.run_name,
                project="olmo3-math-sft",
                cancel_check_interval=10,
                enabled=args.wandb,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        load_path=args.load_path,
    )

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Continue SFT on math data")
    parser.add_argument("--run-name", type=str, default="olmo3-7b-math-sft")
    parser.add_argument("--data-path", type=str, required=True, help="Path to tokenized math data .npy file")
    parser.add_argument("--load-path", type=str, required=True, help="Path to converted OLMo-3-7B-Instruct-SFT checkpoint")
    parser.add_argument("--save-folder", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--work-dir", type=str, default="/tmp/dataset-cache")

    # Training hyperparameters
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=4096, help="Global batch size in tokens")
    parser.add_argument("--microbatch-size", type=int, default=2048, help="Per-rank microbatch size in tokens")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (lower for continued training)")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")

    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)

    if args.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


if __name__ == "__main__":
    main()
