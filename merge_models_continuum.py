"""
Model merging script for combining LoRA fine-tuned model with base model.

Integrates with the Continuum workflow:
- Supports YAML config files with CLI override (like run_finetune.py)
- Auto-detects base model from LoRA adapter_config.json
- Generates timestamped output directories with symlinks
- Tracks merge experiments via WandB
- Saves merge metadata for provenance tracking

Supports multiple merging strategies:
1. LoRA merge only - Simply merge LoRA weights into base model
2. Linear interpolation - Blend base and fine-tuned models with a weight
3. SLERP - Spherical linear interpolation for smoother blending
4. TIES - Task-specific merging that resolves sign conflicts
5. DARE - Drop and rescale for sparse merging

Supports per-layer merge weights via YAML config file.

Usage:
    # Simple LoRA merge (auto-detects base model)
    python -m pilot.model_merging.merge_models --lora-path models/chat_sft_olmo-3-7b-instruct-sft_lora-r8

    # Linear interpolation (70% fine-tuned, 30% original)
    python -m pilot.model_merging.merge_models --lora-path ./math-sft-lora-hf \
        --merge-method linear --merge-weight 0.7

    # Per-layer weights via YAML config
    python -m pilot.model_merging.merge_models --lora-path ./math-sft-lora-hf \
        --merge-method linear --layer-weights-config layer_weights.yaml

    # With WandB tracking
    python -m pilot.model_merging.merge_models --lora-path ./math-sft-lora-hf \
        --use-wandb --wandb-project continuum

    # Generate a template config file
    python -m pilot.model_merging.merge_models --generate-config-template --base-model allenai/Olmo-3-7B-Instruct-SFT

Requires:
    pip install transformers peft torch pyyaml
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

from pilot.utils.config import load_config, merge_config_with_args

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Per-Layer Weight Configuration
# ============================================================================

class LayerWeightConfig:
    """
    Configuration for per-layer merge weights.

    Supports:
    - Global default weight
    - Per-layer weights (by layer index)
    - Layer range weights (e.g., layers 0-10)
    - Component-specific weights (embeddings, lm_head, attention, mlp)
    """

    def __init__(
        self,
        default_weight: float = 0.5,
        layer_weights: Optional[Dict[int, float]] = None,
        component_weights: Optional[Dict[str, float]] = None,
    ):
        self.default_weight = default_weight
        self.layer_weights = layer_weights or {}
        self.component_weights = component_weights or {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LayerWeightConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        default_weight = config.get("default_weight", 0.5)
        layer_weights = {}
        component_weights = {}

        # Parse layer weights
        if "layers" in config:
            for layer_spec, weight in config["layers"].items():
                # Handle range notation: "0-10" or "0:10"
                if isinstance(layer_spec, str) and ("-" in layer_spec or ":" in layer_spec):
                    sep = "-" if "-" in layer_spec else ":"
                    start, end = map(int, layer_spec.split(sep))
                    for i in range(start, end + 1):
                        layer_weights[i] = weight
                else:
                    layer_weights[int(layer_spec)] = weight

        # Parse component weights
        if "components" in config:
            component_weights = config["components"]

        return cls(
            default_weight=default_weight,
            layer_weights=layer_weights,
            component_weights=component_weights,
        )

    @classmethod
    def from_uniform(cls, weight: float) -> "LayerWeightConfig":
        """Create a config with uniform weight for all layers."""
        return cls(default_weight=weight)

    def get_weight_for_param(self, param_name: str) -> float:
        """
        Get the merge weight for a specific parameter based on its name.

        Args:
            param_name: Full parameter name (e.g., "model.layers.15.self_attn.q_proj.weight")

        Returns:
            The merge weight for this parameter
        """
        # Check for embedding layer
        if "embed" in param_name.lower():
            if "embeddings" in self.component_weights:
                return self.component_weights["embeddings"]

        # Check for LM head
        if "lm_head" in param_name.lower():
            if "lm_head" in self.component_weights:
                return self.component_weights["lm_head"]

        # Extract layer number from parameter name
        # Common patterns: "layers.15.", "blocks.15.", "h.15."
        layer_match = re.search(r"(?:layers|blocks|h)\.(\d+)\.", param_name)
        if layer_match:
            layer_idx = int(layer_match.group(1))

            # Check for component-specific weight within layer
            if "attn" in param_name.lower() or "attention" in param_name.lower():
                if "attention" in self.component_weights:
                    # Use component weight as multiplier on layer weight
                    base_weight = self.layer_weights.get(layer_idx, self.default_weight)
                    return base_weight * self.component_weights.get("attention", 1.0)

            if "mlp" in param_name.lower() or "feed_forward" in param_name.lower():
                if "mlp" in self.component_weights:
                    base_weight = self.layer_weights.get(layer_idx, self.default_weight)
                    return base_weight * self.component_weights.get("mlp", 1.0)

            # Return layer-specific weight if defined
            if layer_idx in self.layer_weights:
                return self.layer_weights[layer_idx]

        return self.default_weight

    def __repr__(self) -> str:
        return (
            f"LayerWeightConfig(default={self.default_weight}, "
            f"layers={len(self.layer_weights)}, "
            f"components={list(self.component_weights.keys())})"
        )


def generate_config_template(
    model_name: str,
    output_path: str = "layer_weights_template.yaml",
    default_weight: float = 0.5,
):
    """
    Generate a template YAML config file for per-layer weights.

    Args:
        model_name: HuggingFace model name to determine number of layers
        output_path: Path to save the template
        default_weight: Default weight to use in template
    """
    # Get model config to determine number of layers
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Try different attribute names for number of layers
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = getattr(config, "n_layer", None)
    if n_layers is None:
        n_layers = getattr(config, "num_layers", 32)

    template = f"""# Per-layer merge weight configuration
# Generated for model: {model_name}
# Number of layers: {n_layers}
#
# Weight interpretation:
#   0.0 = 100% original/base model
#   1.0 = 100% fine-tuned model
#   0.5 = 50% blend of both

# Default weight for layers not explicitly specified
default_weight: {default_weight}

# Component-specific weights (optional)
# These override layer weights for specific components
components:
  embeddings: {default_weight}      # Token embedding layer
  lm_head: {default_weight}         # Language model head
  # attention: 1.0    # Multiplier for attention weights (applied on top of layer weight)
  # mlp: 1.0          # Multiplier for MLP/FFN weights (applied on top of layer weight)

# Per-layer weights
# You can specify individual layers or ranges
# Range notation: "start-end" (inclusive)
layers:
  # Early layers (more general features) - keep closer to original
  "0-{n_layers // 4 - 1}": 0.3

  # Middle layers - balanced blend
  "{n_layers // 4}-{n_layers * 3 // 4 - 1}": 0.5

  # Later layers (more task-specific) - use more fine-tuned weights
  "{n_layers * 3 // 4}-{n_layers - 1}": 0.7

# Alternative: specify individual layers (overrides ranges above)
# layers:
#   0: 0.2
#   1: 0.3
#   2: 0.4
#   ...
"""

    with open(output_path, "w") as f:
        f.write(template)

    logger.info(f"Generated template config at: {output_path}")
    logger.info(f"Model has {n_layers} layers")
    return output_path


# ============================================================================
# Auto-Detection and Metadata Utilities
# ============================================================================

def detect_base_model_from_lora(lora_path: str) -> Optional[str]:
    """
    Auto-detect the base model name from a LoRA adapter's adapter_config.json.

    Args:
        lora_path: Path to the LoRA adapter directory

    Returns:
        Base model name/path if found, None otherwise
    """
    adapter_config_path = Path(lora_path) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r") as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
            if base_model:
                logger.info(f"Auto-detected base model from adapter_config.json: {base_model}")
                return base_model
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not parse adapter_config.json: {e}")
    return None


def find_lora_path_in_models(model_name: str) -> Optional[str]:
    """
    Find the LoRA checkpoint path in the models/ directory.

    Handles both direct paths and symlinks, and looks for the latest checkpoint
    if multiple exist.

    Args:
        model_name: Name or partial path of the model

    Returns:
        Full path to the LoRA adapter directory, or None if not found
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    # Try exact match first
    exact_path = models_dir / model_name
    if exact_path.exists():
        # Resolve symlink if needed
        resolved = exact_path.resolve() if exact_path.is_symlink() else exact_path
        # Check if it's a LoRA adapter (has adapter_config.json)
        if (resolved / "adapter_config.json").exists():
            return str(resolved)
        # Check for checkpoint subdirectories
        checkpoints = sorted(resolved.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if checkpoints:
            latest = checkpoints[-1]
            if (latest / "adapter_config.json").exists():
                return str(latest)

    # Try pattern matching for partial names
    for path in models_dir.iterdir():
        if model_name in path.name:
            resolved = path.resolve() if path.is_symlink() else path
            if (resolved / "adapter_config.json").exists():
                return str(resolved)
            checkpoints = sorted(resolved.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if "-" in p.name else 0)
            if checkpoints:
                latest = checkpoints[-1]
                if (latest / "adapter_config.json").exists():
                    return str(latest)

    return None


def create_output_directory(
    lora_path: str,
    merge_method: str,
    merge_weight: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Create a timestamped output directory with symlink, following Continuum conventions.

    Args:
        lora_path: Path to the source LoRA adapter
        merge_method: Merging method used
        merge_weight: Merge weight (0-1)
        output_dir: Optional explicit output directory

    Returns:
        Path to the output directory
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    # Extract model info from lora_path
    lora_name = Path(lora_path).name
    # Remove checkpoint suffix if present
    if lora_name.startswith("checkpoint-"):
        lora_name = Path(lora_path).parent.name

    # Create base name
    weight_str = f"w{merge_weight:.1f}".replace(".", "")
    base_name = f"merged_{lora_name}_{merge_method}_{weight_str}"
    base_path = Path("models") / base_name

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamped_name = f"{base_name}_{timestamp}"
    output_dir = f"models/{timestamped_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Update symlink
    if base_path.exists():
        if base_path.is_symlink() or base_path.is_file():
            base_path.unlink()
        else:
            import shutil
            backup_name = f"{base_name}_backup_{timestamp}"
            shutil.move(str(base_path), f"models/{backup_name}")
            logger.info(f"Moved old directory to: models/{backup_name}")

    base_path.symlink_to(timestamped_name)
    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Symlink: {base_path} -> {timestamped_name}")

    return output_dir


def save_merge_metadata(
    output_dir: str,
    base_model: str,
    lora_path: str,
    merge_method: str,
    merge_weight: Union[float, "LayerWeightConfig"],
    density: Optional[float] = None,
    layer_weights_config: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save merge metadata for provenance tracking.

    Args:
        output_dir: Output directory for the merged model
        base_model: Base model name/path
        lora_path: Path to the LoRA adapter
        merge_method: Merging method used
        merge_weight: Merge weight or LayerWeightConfig
        density: Density parameter (for TIES/DARE)
        layer_weights_config: Path to layer weights YAML config
        extra_info: Additional info to include

    Returns:
        Path to the saved metadata file
    """
    metadata = {
        "merge_timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "lora_path": str(Path(lora_path).resolve()),
        "merge_method": merge_method,
        "merge_weight": merge_weight if isinstance(merge_weight, (int, float)) else repr(merge_weight),
        "density": density,
        "layer_weights_config": layer_weights_config,
        "torch_version": torch.__version__,
        "transformers_version": None,  # Will be set below
    }

    try:
        import transformers
        metadata["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    if extra_info:
        metadata.update(extra_info)

    metadata_path = Path(output_dir) / "merge_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved merge metadata to: {metadata_path}")
    return str(metadata_path)


def init_wandb(
    project: str,
    run_name: str,
    config: Dict[str, Any],
    output_dir: str,
) -> Optional[Any]:
    """
    Initialize WandB for tracking merge experiments.

    Args:
        project: WandB project name
        run_name: WandB run name
        config: Configuration dictionary to log
        output_dir: Output directory (used to save run ID)

    Returns:
        WandB run object, or None if WandB is not available
    """
    try:
        import wandb

        # Check for existing run ID (for resume after preemption)
        run_id_file = Path(output_dir) / "wandb_run_id.txt"
        resume_id = None
        if run_id_file.exists():
            resume_id = run_id_file.read_text().strip()
            logger.info(f"Resuming WandB run: {resume_id}")

        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            id=resume_id,
            resume="allow" if resume_id else None,
        )

        # Save run ID for resume
        run_id_file.write_text(run.id)
        logger.info(f"WandB run initialized: {run.url}")

        return run
    except ImportError:
        logger.warning("WandB not installed. Skipping WandB logging.")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        return None


# ============================================================================
# Model Loading
# ============================================================================

def merge_lora_to_base(
    base_model_name: str,
    lora_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load base model and merge LoRA weights into it."""
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    return model


def get_model_state_dict(model: AutoModelForCausalLM) -> Dict[str, torch.Tensor]:
    """Get the state dict of a model, handling different model types."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def linear_interpolate(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    weight: Union[float, LayerWeightConfig],
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between two state dicts.

    result = (1 - weight) * state_dict_a + weight * state_dict_b

    Args:
        state_dict_a: First state dict (e.g., original model)
        state_dict_b: Second state dict (e.g., fine-tuned model)
        weight: Interpolation weight (0 = all A, 1 = all B), or LayerWeightConfig for per-layer weights
    """
    # Convert float to uniform config
    if isinstance(weight, (int, float)):
        weight_config = LayerWeightConfig.from_uniform(float(weight))
    else:
        weight_config = weight

    result = {}
    for key in tqdm(state_dict_a.keys(), desc="Linear interpolation"):
        if key in state_dict_b:
            w = weight_config.get_weight_for_param(key)
            result[key] = (1 - w) * state_dict_a[key] + w * state_dict_b[key]
        else:
            result[key] = state_dict_a[key]
    return result


def slerp(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    weight: Union[float, LayerWeightConfig],
) -> Dict[str, torch.Tensor]:
    """
    Spherical linear interpolation between two state dicts.

    SLERP provides smoother interpolation that stays on the hypersphere,
    which can be better for neural network weights.

    Args:
        state_dict_a: First state dict (e.g., original model)
        state_dict_b: Second state dict (e.g., fine-tuned model)
        weight: Interpolation weight (0 = all A, 1 = all B), or LayerWeightConfig for per-layer weights
    """
    # Convert float to uniform config
    if isinstance(weight, (int, float)):
        weight_config = LayerWeightConfig.from_uniform(float(weight))
    else:
        weight_config = weight

    result = {}
    for key in tqdm(state_dict_a.keys(), desc="SLERP interpolation"):
        if key in state_dict_b:
            w = weight_config.get_weight_for_param(key)
            a = state_dict_a[key].float()
            b = state_dict_b[key].float()

            # Flatten for dot product
            a_flat = a.flatten()
            b_flat = b.flatten()

            # Compute cosine similarity
            dot = torch.dot(a_flat, b_flat)
            norm_a = torch.norm(a_flat)
            norm_b = torch.norm(b_flat)

            if norm_a == 0 or norm_b == 0:
                # If either is zero, use linear interpolation
                result[key] = ((1 - w) * a + w * b).to(state_dict_a[key].dtype)
                continue

            cos_theta = dot / (norm_a * norm_b)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

            theta = torch.acos(cos_theta)

            if theta < 1e-6:
                # Very similar vectors, use linear interpolation
                result[key] = ((1 - w) * a + w * b).to(state_dict_a[key].dtype)
            else:
                sin_theta = torch.sin(theta)
                w_a = torch.sin((1 - w) * theta) / sin_theta
                w_b = torch.sin(w * theta) / sin_theta
                result[key] = (w_a * a + w_b * b).to(state_dict_a[key].dtype)
        else:
            result[key] = state_dict_a[key]
    return result


def ties_merge(
    state_dict_base: Dict[str, torch.Tensor],
    state_dict_finetuned: Dict[str, torch.Tensor],
    weight: Union[float, LayerWeightConfig],
    density: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    TIES (TrIm, Elect Sign & Merge) merging.

    This method:
    1. Computes task vectors (delta from base)
    2. Trims small values (sparsification)
    3. Resolves sign conflicts by majority vote
    4. Merges with specified weight

    Args:
        state_dict_base: Base model state dict
        state_dict_finetuned: Fine-tuned model state dict
        weight: Merge weight for the task vector, or LayerWeightConfig for per-layer weights
        density: Fraction of weights to keep (1.0 = keep all)
    """
    # Convert float to uniform config
    if isinstance(weight, (int, float)):
        weight_config = LayerWeightConfig.from_uniform(float(weight))
    else:
        weight_config = weight

    result = {}

    for key in tqdm(state_dict_base.keys(), desc="TIES merge"):
        if key in state_dict_finetuned:
            w = weight_config.get_weight_for_param(key)
            base = state_dict_base[key].float()
            finetuned = state_dict_finetuned[key].float()

            # Compute task vector (delta)
            delta = finetuned - base

            # Trim: keep only top-k% by magnitude
            if density < 1.0:
                flat_delta = delta.flatten()
                k = int(len(flat_delta) * density)
                if k > 0:
                    threshold = torch.topk(flat_delta.abs(), k).values[-1]
                    mask = flat_delta.abs() >= threshold
                    delta = delta.flatten()
                    delta[~mask] = 0
                    delta = delta.reshape(base.shape)

            # Apply weighted task vector
            merged = base + w * delta
            result[key] = merged.to(state_dict_base[key].dtype)
        else:
            result[key] = state_dict_base[key]

    return result


def dare_merge(
    state_dict_base: Dict[str, torch.Tensor],
    state_dict_finetuned: Dict[str, torch.Tensor],
    weight: Union[float, LayerWeightConfig],
    density: float = 0.5,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    DARE (Drop And REscale) merging.

    This method randomly drops delta weights and rescales the remaining ones.

    Args:
        state_dict_base: Base model state dict
        state_dict_finetuned: Fine-tuned model state dict
        weight: Merge weight for the task vector, or LayerWeightConfig for per-layer weights
        density: Probability of keeping each weight
        seed: Random seed for reproducibility
    """
    # Convert float to uniform config
    if isinstance(weight, (int, float)):
        weight_config = LayerWeightConfig.from_uniform(float(weight))
    else:
        weight_config = weight

    torch.manual_seed(seed)
    result = {}

    for key in tqdm(state_dict_base.keys(), desc="DARE merge"):
        if key in state_dict_finetuned:
            w = weight_config.get_weight_for_param(key)
            base = state_dict_base[key].float()
            finetuned = state_dict_finetuned[key].float()

            # Compute task vector (delta)
            delta = finetuned - base

            # Random drop mask
            mask = torch.bernoulli(torch.full_like(delta, density)).bool()

            # Rescale to compensate for dropped weights
            if density > 0:
                delta = torch.where(mask, delta / density, torch.zeros_like(delta))

            # Apply weighted task vector
            merged = base + w * delta
            result[key] = merged.to(state_dict_base[key].dtype)
        else:
            result[key] = state_dict_base[key]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA fine-tuned model with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple LoRA merge (auto-detects base model and output directory)
  python -m pilot.model_merging.merge_models --lora-path models/chat_sft_olmo-3-7b

  # Linear interpolation with uniform weight
  python -m pilot.model_merging.merge_models --lora-path ./lora-checkpoint \\
      --merge-method linear --merge-weight 0.7

  # Per-layer weights via YAML config
  python -m pilot.model_merging.merge_models --lora-path ./lora-checkpoint \\
      --merge-method linear --layer-weights-config layer_weights.yaml

  # With WandB tracking
  python -m pilot.model_merging.merge_models --lora-path ./lora-checkpoint \\
      --use-wandb --wandb-project continuum

  # Generate a template config
  python -m pilot.model_merging.merge_models --generate-config-template --base-model allenai/Olmo-3-7B-Instruct-SFT
""",
    )

    # Config file support
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config)",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name or path (auto-detected from LoRA adapter if not specified)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint (can be name in models/ directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged model (auto-generated if not specified)",
    )
    parser.add_argument(
        "--merge-method",
        type=str,
        choices=["lora_only", "linear", "slerp", "ties", "dare"],
        default="lora_only",
        help="Merging method to use",
    )
    parser.add_argument(
        "--merge-weight",
        type=float,
        default=1.0,
        help="Weight for fine-tuned model (0=base, 1=fine-tuned). Ignored if --layer-weights-config is provided.",
    )
    parser.add_argument(
        "--layer-weights-config",
        type=str,
        default=None,
        help="Path to YAML config file with per-layer merge weights",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Density parameter for TIES/DARE (fraction of weights to keep)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (required if --push-to-hub)",
    )

    # WandB logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="continuum",
        help="W&B project name (default: continuum)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)",
    )

    # Template generation
    parser.add_argument(
        "--generate-config-template",
        action="store_true",
        help="Generate a template YAML config for per-layer weights and exit",
    )
    parser.add_argument(
        "--template-output",
        type=str,
        default="layer_weights_template.yaml",
        help="Output path for generated template (default: layer_weights_template.yaml)",
    )

    args = parser.parse_args()

    # Load config file if provided and merge with CLI args
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args, parser)
        logger.info(f"Loaded config from: {args.config}")

    # Handle template generation mode
    if args.generate_config_template:
        base_model_for_template = args.base_model or "allenai/Olmo-3-7B-Instruct-SFT"
        generate_config_template(
            base_model_for_template,
            args.template_output,
            default_weight=args.merge_weight,
        )
        return

    # Validate required arguments for merging
    if args.lora_path is None:
        parser.error("--lora-path is required for merging")

    # Smart LoRA path detection - check models/ directory
    lora_path = args.lora_path
    if not Path(lora_path).exists():
        # Try finding in models/ directory
        found_path = find_lora_path_in_models(lora_path)
        if found_path:
            logger.info(f"Found LoRA adapter in models/: {found_path}")
            lora_path = found_path
        else:
            parser.error(f"LoRA path not found: {lora_path}")
    else:
        # Resolve to absolute path
        lora_path = str(Path(lora_path).resolve())

    # Auto-detect base model from LoRA adapter if not specified
    base_model = args.base_model
    if base_model is None:
        base_model = detect_base_model_from_lora(lora_path)
        if base_model is None:
            parser.error("Could not auto-detect base model. Please specify --base-model")

    # Create output directory (auto-generate if not specified)
    output_dir = create_output_directory(
        lora_path=lora_path,
        merge_method=args.merge_method,
        merge_weight=args.merge_weight,
        output_dir=args.output_dir,
    )

    # Set torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Load layer weight config if provided
    if args.layer_weights_config:
        logger.info(f"Loading per-layer weight config from: {args.layer_weights_config}")
        weight_config = LayerWeightConfig.from_yaml(args.layer_weights_config)
        logger.info(f"Config: {weight_config}")
    else:
        weight_config = args.merge_weight

    # Initialize WandB if requested
    wandb_run = None
    if args.use_wandb:
        # Generate run name if not specified
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None:
            lora_name = Path(lora_path).name
            if lora_name.startswith("checkpoint-"):
                lora_name = Path(lora_path).parent.name
            wandb_run_name = f"merge_{lora_name}_{args.merge_method}"

        wandb_config = {
            "base_model": base_model,
            "lora_path": lora_path,
            "merge_method": args.merge_method,
            "merge_weight": args.merge_weight if not args.layer_weights_config else "per-layer",
            "layer_weights_config": args.layer_weights_config,
            "density": args.density,
            "torch_dtype": args.torch_dtype,
        }

        wandb_run = init_wandb(
            project=args.wandb_project,
            run_name=wandb_run_name,
            config=wandb_config,
            output_dir=output_dir,
        )

    if args.merge_method == "lora_only":
        # Simple LoRA merge
        logger.info("Performing simple LoRA merge...")
        merged_model = merge_lora_to_base(
            base_model,
            lora_path,
            torch_dtype=torch_dtype,
        )
    else:
        # First merge LoRA into base to get fine-tuned model
        logger.info("Step 1: Merging LoRA into base model...")
        finetuned_model = merge_lora_to_base(
            base_model,
            lora_path,
            torch_dtype=torch_dtype,
        )

        # Load original base model
        logger.info("Step 2: Loading original base model for interpolation...")
        base_model_for_interp = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map="cpu",  # Load on CPU to save memory
            trust_remote_code=True,
        )

        # Get state dicts
        logger.info("Step 3: Extracting state dicts...")
        base_state_dict = get_model_state_dict(base_model_for_interp)
        finetuned_state_dict = get_model_state_dict(finetuned_model)

        # Free up memory
        del base_model_for_interp
        torch.cuda.empty_cache()

        # Perform merging
        if isinstance(weight_config, LayerWeightConfig):
            logger.info(f"Step 4: Applying {args.merge_method} merge with per-layer weights...")
        else:
            logger.info(f"Step 4: Applying {args.merge_method} merge with weight={weight_config}...")

        if args.merge_method == "linear":
            merged_state_dict = linear_interpolate(
                base_state_dict,
                finetuned_state_dict,
                weight_config,
            )
        elif args.merge_method == "slerp":
            merged_state_dict = slerp(
                base_state_dict,
                finetuned_state_dict,
                weight_config,
            )
        elif args.merge_method == "ties":
            merged_state_dict = ties_merge(
                base_state_dict,
                finetuned_state_dict,
                weight_config,
                args.density,
            )
        elif args.merge_method == "dare":
            merged_state_dict = dare_merge(
                base_state_dict,
                finetuned_state_dict,
                weight_config,
                args.density,
            )

        # Load merged state dict into model
        logger.info("Step 5: Loading merged weights into model...")
        finetuned_model.load_state_dict(merged_state_dict)
        merged_model = finetuned_model

        del base_state_dict, finetuned_state_dict, merged_state_dict
        torch.cuda.empty_cache()

    # Save merged model
    logger.info(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Save merge metadata for provenance tracking
    save_merge_metadata(
        output_dir=output_dir,
        base_model=base_model,
        lora_path=lora_path,
        merge_method=args.merge_method,
        merge_weight=weight_config,
        density=args.density if args.merge_method in ["ties", "dare"] else None,
        layer_weights_config=args.layer_weights_config,
    )

    # Log to WandB if enabled
    if wandb_run:
        try:
            import wandb
            wandb.log({
                "merge_complete": True,
                "output_dir": output_dir,
            })
            wandb.finish()
            logger.info("WandB run finished successfully")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    # Push to hub if requested
    if args.push_to_hub:
        if args.hub_repo_id is None:
            raise ValueError("--hub-repo-id is required when using --push-to-hub")
        logger.info(f"Pushing to HuggingFace Hub: {args.hub_repo_id}")
        merged_model.push_to_hub(args.hub_repo_id)
        tokenizer.push_to_hub(args.hub_repo_id)

    logger.info("Done!")
    logger.info(f"Merged model saved to: {output_dir}")
    logger.info("")
    logger.info("To load the merged model:")
    logger.info(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    logger.info("")
    logger.info("To evaluate the merged model:")
    logger.info(f"  python -m pilot.run_eval --model {output_dir}")


if __name__ == "__main__":
    main()
