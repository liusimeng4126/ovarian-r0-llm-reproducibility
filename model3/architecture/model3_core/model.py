"""Small shared-site model for the 14-site R0 prediction task.

The model consumes *already transformed* features. Fit preprocessing on training
patients only. This module does not infer missingness, normalise inputs, choose a
threshold, or apply a sigmoid to the returned logits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """All dimensions needed to reconstruct a model from a saved checkpoint."""

    site_numeric_dim: int
    global_numeric_dim: int
    hidden_dim: int = 32
    dropout: float = 0.2
    category_embedding_dim: int = 4
    site_embedding_dim: int = 8
    n_sites: int = 14
    status_vocab_size: int = 6
    change_vocab_size: int = 9
    global_categorical_dim: int = 4
    use_attention: bool = True
    use_site_identity: bool = True

    def __post_init__(self) -> None:
        positive_dims = (
            "site_numeric_dim",
            "global_numeric_dim",
            "hidden_dim",
            "category_embedding_dim",
            "site_embedding_dim",
            "status_vocab_size",
            "change_vocab_size",
            "global_categorical_dim",
        )
        for name in positive_dims:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.n_sites != 14:
            raise ValueError("This study requires exactly 14 fixed anatomical sites.")
        if self.global_categorical_dim != 4:
            raise ValueError("The input schema requires 4 global categorical features.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if type(self.use_attention) is not bool:
            raise ValueError("use_attention must be boolean.")
        if type(self.use_site_identity) is not bool:
            raise ValueError("use_site_identity must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return cls(**dict(values))


class Model3(nn.Module):
    """One site encoder, explicit site identity, attention, and a global branch.

    Required batch fields:
        site_numeric:       float32, (B, 14, site_numeric_dim)
        site_categorical:   int64,   (B, 14, 3), PRE/POST/change in that order
        global_numeric:     float32, (B, global_numeric_dim)
        global_categorical: int64,   (B, 4), all using the status vocabulary

    Every one of the 14 sites participates in attention. In particular,
    ``not_mentioned`` is an observed category, not an attention padding mask.
    Missing numeric values must have been handled by training-fitted
    preprocessing and accompanied by the supplied missingness indicators.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.status_embedding = nn.Embedding(
            config.status_vocab_size, config.category_embedding_dim
        )
        self.change_embedding = nn.Embedding(
            config.change_vocab_size, config.category_embedding_dim
        )
        self.site_embedding = nn.Embedding(config.n_sites, config.site_embedding_dim)
        # This is the only site encoder. nn.Linear operates on the last axis,
        # so the same weights transform every patient and every anatomical site.
        site_input_dim = (
            config.site_numeric_dim
            + 3 * config.category_embedding_dim
            + config.site_embedding_dim
        )
        self.site_encoder = nn.Sequential(
            nn.Linear(site_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        attention_hidden = max(8, config.hidden_dim // 2)
        self.attention_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        global_input_dim = (
            config.global_numeric_dim
            + config.global_categorical_dim * config.category_embedding_dim
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.register_buffer(
            "site_indices", torch.arange(config.n_sites, dtype=torch.long)
        )

    def _validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        required = (
            "site_numeric",
            "site_categorical",
            "global_numeric",
            "global_categorical",
        )
        for name in required:
            if name not in batch:
                raise ValueError(f"Missing required batch field: {name}")
            if not isinstance(batch[name], torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")

        c = self.config
        site_numeric = batch["site_numeric"]
        if site_numeric.ndim != 3:
            raise ValueError("site_numeric must have shape (B, 14, site_numeric_dim).")
        batch_size = site_numeric.shape[0]
        if batch_size == 0:
            raise ValueError("Empty batches are not supported.")
        expected_shapes = {
            "site_numeric": (batch_size, c.n_sites, c.site_numeric_dim),
            "site_categorical": (batch_size, c.n_sites, 3),
            "global_numeric": (batch_size, c.global_numeric_dim),
            "global_categorical": (batch_size, c.global_categorical_dim),
        }
        for name, expected in expected_shapes.items():
            if tuple(batch[name].shape) != expected:
                raise ValueError(
                    f"{name} has shape {tuple(batch[name].shape)}; expected {expected}."
                )
            if batch[name].device != self.site_indices.device:
                raise ValueError(
                    f"{name} is on {batch[name].device}, but model is on "
                    f"{self.site_indices.device}; move both to the same device."
                )
        model_dtype = self.site_encoder[0].weight.dtype
        for name in ("site_numeric", "global_numeric"):
            if batch[name].dtype != model_dtype:
                raise TypeError(
                    f"{name} must have dtype {model_dtype}; got {batch[name].dtype}."
                )
        for name in ("site_categorical", "global_categorical"):
            if batch[name].dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.int64.")
        category_fields = (
            ("site PRE/POST status", batch["site_categorical"][..., :2], c.status_vocab_size),
            ("site change", batch["site_categorical"][..., 2], c.change_vocab_size),
            ("global status", batch["global_categorical"], c.status_vocab_size),
        )
        for name, values, vocabulary_size in category_fields:
            if bool(torch.any((values < 0) | (values >= vocabulary_size))):
                raise ValueError(
                    f"{name} category IDs must be in [0, {vocabulary_size - 1}]."
                )

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_batch(batch)
        numeric = batch["site_numeric"]
        categories = batch["site_categorical"]
        batch_size = numeric.shape[0]
        pre_post = self.status_embedding(categories[..., :2]).flatten(start_dim=-2)
        changes = self.change_embedding(categories[..., 2])
        identities = self.site_embedding(self.site_indices).unsqueeze(0).expand(batch_size, -1, -1)
        if not self.config.use_site_identity:
            identities = torch.zeros_like(identities)
        sites = self.site_encoder(
            torch.cat((numeric, pre_post, changes, identities), dim=-1)
        )
        if self.config.use_attention:
            attention = torch.softmax(self.attention_scorer(sites).squeeze(-1), dim=1)
        else:
            attention = torch.full(
                (batch_size, self.config.n_sites),
                1.0 / self.config.n_sites,
                dtype=sites.dtype,
                device=sites.device,
            )
        patient_sites = torch.sum(sites * attention.unsqueeze(-1), dim=1)
        global_categories = self.status_embedding(
            batch["global_categorical"]
        ).flatten(start_dim=1)
        global_features = self.global_encoder(
            torch.cat((batch["global_numeric"], global_categories), dim=1)
        )
        logits = self.classifier(
            torch.cat((patient_sites, global_features), dim=1)
        ).squeeze(-1)
        return logits, attention
