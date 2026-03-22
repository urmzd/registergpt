"""Base class for all AGI model variants."""
from __future__ import annotations

import abc
from typing import ClassVar

import torch.nn as nn
from pydantic_settings import BaseSettings
from torch import Tensor


class CommonSettings(BaseSettings):
    """Shared hyperparameters across most model variants."""
    vocab_size: int = 1024
    num_steps: int = 8
    n_fourier_basis: int = 16
    n_channels: int = 128
    activation: str = "gelu"
    logit_softcap: float = 30.0
    decay_init: float = 3.0


class AgiModel(nn.Module, abc.ABC):
    """Abstract base for all model variants.

    Subclasses must define class-level metadata, a Settings inner class,
    and implement forward(). The registry auto-discovers any AgiModel
    subclass that has a `version` set.
    """

    # -- Metadata (override in each subclass) --
    version: ClassVar[str] = ""
    architecture: ClassVar[str] = ""
    cross_position: ClassVar[str] = ""
    within_position: ClassVar[str] = ""

    # -- Config (override with a Settings subclass) --
    Settings: ClassVar[type[CommonSettings]] = CommonSettings

    @classmethod
    def build_kwargs(cls, args) -> dict:
        """Extract constructor kwargs from a config/args namespace.

        Default: instantiate Settings from the args namespace.
        Override if the mapping between args and __init__ params is non-trivial.
        """
        fields = cls.Settings.model_fields
        return {k: getattr(args, k) for k in fields if hasattr(args, k)}

    @abc.abstractmethod
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Return scalar loss."""
        ...
