"""Helpers for reading likelihood and diffusion metadata from ``ModelSpec``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def get_per_variable_diffusion(spec: ModelSpec) -> list[DistributionFamily]:
    """Return the canonical per-variable diffusion noise families."""
    return list(numeric.diffusion_families(spec))


def has_student_t_diffusion(spec: ModelSpec) -> bool:
    """Return whether any latent process uses Student-t diffusion noise."""
    from nof1_causal_lab.artifacts.likelihood import DistributionFamily

    return DistributionFamily.STUDENT_T in set(get_per_variable_diffusion(spec))


def get_per_channel_manifest(spec: ModelSpec) -> list[DistributionFamily]:
    """Return the canonical per-channel observation noise families."""
    return list(numeric.observation_families(spec))


def get_per_channel_links(spec: ModelSpec) -> list[LinkFunction]:
    """Resolve per-channel link functions."""
    return numeric.observation_links(spec)
