# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility helpers to import all modules in a package recursively."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None) -> None:
    """Import all sub-packages in a package recursively."""
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    package = importlib.import_module(package_name)
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yield module info recursively while skipping blacklisted packages."""
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(package_path: str, memory: dict[str, bool] = {}) -> bool:
        if package_path in memory:
            return True
        memory[package_path] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        if any(black_pkg_name in info.name for black_pkg_name in blacklist_pkgs):
            continue

        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                child_path: list[str] = getattr(sys.modules[info.name], "__path__", [])
                child_path = [item for item in child_path if not seen(item)]
                yield from _walk_packages(child_path, info.name + ".", onerror, blacklist_pkgs)
