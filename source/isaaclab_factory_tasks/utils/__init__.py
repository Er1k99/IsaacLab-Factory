"""Utility helpers for config loading and task registration."""

from .importer import import_packages


def hydra_task_config(*args, **kwargs):
    """Import Hydra helpers lazily so package registration does not require Hydra."""
    from .hydra import hydra_task_config as _hydra_task_config

    return _hydra_task_config(*args, **kwargs)


def load_cfg_from_registry(*args, **kwargs):
    """Import registry helpers lazily to avoid importing IsaacLab at package import time."""
    from .parse_cfg import load_cfg_from_registry as _load_cfg_from_registry

    return _load_cfg_from_registry(*args, **kwargs)


def parse_env_cfg(*args, **kwargs):
    """Import env config helpers lazily to avoid importing IsaacLab at package import time."""
    from .parse_cfg import parse_env_cfg as _parse_env_cfg

    return _parse_env_cfg(*args, **kwargs)


def get_checkpoint_path(*args, **kwargs):
    """Import checkpoint helpers lazily to avoid importing IsaacLab at package import time."""
    from .parse_cfg import get_checkpoint_path as _get_checkpoint_path

    return _get_checkpoint_path(*args, **kwargs)
