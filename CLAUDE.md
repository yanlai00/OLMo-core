# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OLMo-core is the core training infrastructure library for the Open Language Model (OLMo) project by Allen Institute for AI. It provides building blocks for large language model training and inference, including neural network modules, distributed training, data loading, optimization, and checkpointing.

## Development Commands

```bash
# Install in development mode
pip install -e '.[all]'    # or: uv sync --all-extras

# Code formatting
make style                  # Format code with isort and black
make style-check            # Check formatting without changes

# Linting and type checking
make lint-check             # Run ruff linter
make type-check             # Run mypy type checker
make checks                 # Run all checks (style, lint, type)

# Testing
pytest -v src/test                      # Run all tests
pytest src/test/nn/rope_test.py         # Run specific test file
pytest -k rope                          # Run tests matching keyword
pytest -m gpu                           # Run only GPU tests
pytest -m "not gpu"                     # Exclude GPU tests

# Documentation
make docs                   # Build and serve docs locally
```

## Code Architecture

### Directory Structure
- `src/olmo_core/` - Main library code
- `src/test/` - Tests (mirrors library structure)
- `src/scripts/official/` - Official training scripts for OLMo-2 and OLMo-3
- `src/examples/` - Example code and reference implementations

### Key Modules
- `nn/` - Neural network building blocks: attention (flash-attn, ring-attn), transformer blocks, MoE, RoPE embeddings
- `train/` - Training orchestration: Trainer class, TrainModule abstraction, callbacks system
- `data/` - Data loading: NumPy datasets (memory-mapped), composable transforms, source mixing
- `optim/` - Optimizers (AdamW, Lion) and LR schedulers (cosine, linear)
- `distributed/` - Distributed training: checkpoint sharding, parallel strategies, collective communications
- `launch/` - Job launching (Beaker integration)

### Configuration Pattern
Everything uses dataclass-based configs inheriting from `Config` base class. Configs support:
- YAML loading via `Config.from_yaml()`
- Command-line overrides of nested values (e.g., `--train_module.optim.lr=6e-3`)
- OmegaConf for config merging

### Training Pipeline Flow
```
TrainerConfig → Trainer → TrainModule → Model + Optimizer
                  ↓
              Callbacks (checkpointing, evaluation, logging)
                  ↓
              DataLoader → NumPy datasets
```

## Testing Conventions

- Test files: `*_test.py` (not `test_*.py`)
- Test functions: `test_*`
- Mirror source structure: `src/olmo_core/nn/rope.py` → `src/test/nn/rope_test.py`
- GPU tests use `@requires_gpu` decorator (auto-skip on CPU)
- Multi-GPU tests use `@requires_multi_gpu` and `run_distributed_test()` helper
- Prefer `pytest.mark.parametrize` for test variations

## Running Training Scripts

Official training scripts are launched with torchrun:
```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/checkpoints
```

## PR Requirements

- Update `CHANGELOG.md` (enforced by CI)
- All checks must pass (`make checks`)
- Add tests for new functionality
- Update docstrings for new public APIs
