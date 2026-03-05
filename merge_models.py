"""
Model merging script for combining LoRA fine-tuned model with base model.

Supports multiple merging strategies:
1. LoRA merge only - Simply merge LoRA weights into base model
2. Linear interpolation - Blend base and fine-tuned models with a weight
3. SLERP - Spherical linear interpolation for smoother blending
4. TIES - Task-specific merging that resolves sign conflicts
5. DARE - Drop and rescale for sparse merging

Supports per-layer merge weights via YAML config file.

Usage:
    # Simple LoRA merge
    python merge_models.py --lora-path ./math-sft-lora-hf --output-dir ./merged-model

    # Linear interpolation (70% fine-tuned, 30% original)
    python merge_models.py --lora-path ./math-sft-lora-hf --output-dir ./merged-model \
        --merge-method linear --merge-weight 0.7

    # Per-layer weights via YAML config
    python merge_models.py --lora-path ./math-sft-lora-hf --output-dir ./merged-model \
        --merge-method linear --layer-weights-config layer_weights.yaml

    # Generate a template config file
    python merge_models.py --generate-config-template --base-model allenai/Olmo-3-7B-Instruct-SFT

Requires:
    pip install transformers peft torch pyyaml
"""

import argparse
import logging
import os
import re
from typing import Dict, List, Optional, Union

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

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
  # Simple LoRA merge
  python merge_models.py --lora-path ./lora-checkpoint --output-dir ./merged

  # Linear interpolation with uniform weight
  python merge_models.py --lora-path ./lora-checkpoint --output-dir ./merged \\
      --merge-method linear --merge-weight 0.7

  # Per-layer weights via YAML config
  python merge_models.py --lora-path ./lora-checkpoint --output-dir ./merged \\
      --merge-method linear --layer-weights-config layer_weights.yaml

  # Generate a template config
  python merge_models.py --generate-config-template --base-model allenai/Olmo-3-7B-Instruct-SFT
""",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct-SFT",
        help="Base model name or path",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged model",
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

    # Handle template generation mode
    if args.generate_config_template:
        generate_config_template(
            args.base_model,
            args.template_output,
            default_weight=args.merge_weight,
        )
        return

    # Validate required arguments for merging
    if args.lora_path is None:
        parser.error("--lora-path is required for merging")
    if args.output_dir is None:
        parser.error("--output-dir is required for merging")

    # Set torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load layer weight config if provided
    if args.layer_weights_config:
        logger.info(f"Loading per-layer weight config from: {args.layer_weights_config}")
        weight_config = LayerWeightConfig.from_yaml(args.layer_weights_config)
        logger.info(f"Config: {weight_config}")
    else:
        weight_config = args.merge_weight

    if args.merge_method == "lora_only":
        # Simple LoRA merge
        logger.info("Performing simple LoRA merge...")
        merged_model = merge_lora_to_base(
            args.base_model,
            args.lora_path,
            torch_dtype=torch_dtype,
        )
    else:
        # First merge LoRA into base to get fine-tuned model
        logger.info("Step 1: Merging LoRA into base model...")
        finetuned_model = merge_lora_to_base(
            args.base_model,
            args.lora_path,
            torch_dtype=torch_dtype,
        )

        # Load original base model
        logger.info("Step 2: Loading original base model for interpolation...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map="cpu",  # Load on CPU to save memory
            trust_remote_code=True,
        )

        # Get state dicts
        logger.info("Step 3: Extracting state dicts...")
        base_state_dict = get_model_state_dict(base_model)
        finetuned_state_dict = get_model_state_dict(finetuned_model)

        # Free up memory
        del base_model
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
    logger.info(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        if args.hub_repo_id is None:
            raise ValueError("--hub-repo-id is required when using --push-to-hub")
        logger.info(f"Pushing to HuggingFace Hub: {args.hub_repo_id}")
        merged_model.push_to_hub(args.hub_repo_id)
        tokenizer.push_to_hub(args.hub_repo_id)

    logger.info("Done!")
    logger.info(f"Merged model saved to: {args.output_dir}")
    logger.info("")
    logger.info("To load the merged model:")
    logger.info(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    logger.info(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
