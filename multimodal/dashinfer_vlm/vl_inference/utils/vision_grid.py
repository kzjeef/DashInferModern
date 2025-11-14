# Copyright (c) 2026 Segno System.
"""Validation helpers for vision patch-grid metadata."""

from operator import index


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_vision_grid(grid_thw, spatial_merge_size=2):
    """Validate and return a ``(time, height, width)`` vision grid."""
    if grid_thw is None:
        raise ValueError("grid_thw is required")
    try:
        values = tuple(grid_thw)
    except TypeError as exc:
        raise ValueError("grid_thw must contain [t, h, w]") from exc
    if len(values) != 3:
        raise ValueError("grid_thw must contain [t, h, w]")

    merge_size = _positive_integer(spatial_merge_size, "spatial_merge_size")
    time, height, width = (
        _positive_integer(value, name)
        for value, name in zip(values, ("grid_t", "grid_h", "grid_w"))
    )
    if height % merge_size or width % merge_size:
        raise ValueError(
            "grid height and width must be divisible by spatial_merge_size")
    return time, height, width


def vision_token_count(grid_thw, spatial_merge_size=2):
    """Return the language-token span occupied by one vision grid."""
    time, height, width = normalize_vision_grid(
        grid_thw, spatial_merge_size=spatial_merge_size)
    return time * (height // spatial_merge_size) * (
        width // spatial_merge_size)
