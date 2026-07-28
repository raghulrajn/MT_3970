"""Dummy model for testing the training pipeline.

This model demonstrates how to:
1. Read architecture parameters from config
2. Build a dynamic MLP based on config
3. Use the base class for automatic metric tracking

It creates synthetic data to test the pipeline without requiring real data processing.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base_model import BaseSurrogateModel


class DummyModel(BaseSurrogateModel):
    """Dummy model that demonstrates config usage and pipeline testing.

    Shows students how to:
    - Read architecture config (input_dim, hidden_dims, output_dim)
    - Build layers dynamically from config
    - Use training hyperparameters from config

    Uses base class training/validation/test steps for automatic metric logging.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Read architecture from config
        arch = config.get("architecture", {})
        self.input_dim = arch.get("input_dim", 10)
        self.hidden_dims: List[int] = arch.get("hidden_dims", [128, 64, 32])
        self.output_dim = arch.get("output_dim", 3)
        activation = arch.get("activation", "relu")
        dropout = arch.get("dropout", 0.1)
        use_batch_norm = arch.get("batch_norm", True)

        # Build MLP dynamically from config
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.model = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer from config."""
        optimizer_cfg = self.config.get("optimizer", {})
        optimizer_name = optimizer_cfg.get("name", "adam").lower()
        lr = optimizer_cfg.get("learning_rate", 0.001)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0001)

        if optimizer_name == "adam":
            betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
            return torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
            )
        elif optimizer_name == "adamw":
            betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
            return torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
            )
        elif optimizer_name == "sgd":
            momentum = optimizer_cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def preprocess_data(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert batch to (inputs, targets) format.

        Creates synthetic data matching config dimensions for pipeline testing.
        Real models would extract actual data from the batch here.
        """
        # Determine batch size from actual data
        if isinstance(batch, dict):
            if "parameters" in batch:
                params = batch["parameters"]
                bs = params.shape[0] if isinstance(params, torch.Tensor) else 1
            elif "blank" in batch:
                blank = batch["blank"]
                bs = max(1, blank.x.shape[0] // 2048) if hasattr(blank, "x") else 1
            else:
                bs = 1
        elif isinstance(batch, (tuple, list)) and len(batch) >= 1:
            bs = batch[0].shape[0] if hasattr(batch[0], "shape") else 1
        else:
            bs = 1

        # Generate synthetic data matching config dimensions
        # Real models would process actual batch data here
        inputs = torch.randn(bs, self.input_dim, device=self.device)
        targets = torch.full((bs, self.output_dim), 0.5, device=self.device)
        return inputs, targets
