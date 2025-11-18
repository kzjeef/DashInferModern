# Copyright (c) 2026 Segno System.
"""Dependency-free MRoPE position construction for vision prompts."""

from .vision_grid import normalize_vision_grid, vision_token_count


def _append_text_positions(positions, start, length):
    values = range(start, start + length)
    for axis in positions:
        axis.extend(values)


def build_mrope_positions(
    input_tokens,
    vision_grids,
    image_token_id,
    spatial_merge_size=2,
):
    """Build three-axis Qwen vision positions for one token sequence."""
    tokens = list(input_tokens)
    grids = list(vision_grids)
    positions = [[], [], []]
    token_cursor = 0
    position_cursor = 0

    for grid_index, grid in enumerate(grids):
        time, height, width = normalize_vision_grid(
            grid, spatial_merge_size=spatial_merge_size)
        token_count = vision_token_count(
            (time, height, width), spatial_merge_size=spatial_merge_size)
        try:
            image_start = tokens.index(image_token_id, token_cursor)
        except ValueError as exc:
            raise ValueError(
                f"vision grid {grid_index} has no image token span") from exc

        text_length = image_start - token_cursor
        _append_text_positions(positions, position_cursor, text_length)
        position_cursor += text_length

        image_end = image_start + token_count
        if image_end > len(tokens) or any(
            token != image_token_id for token in tokens[image_start:image_end]
        ):
            raise ValueError(
                f"vision grid {grid_index} expects {token_count} contiguous "
                "image tokens")

        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        for time_index in range(time):
            for height_index in range(merged_height):
                for width_index in range(merged_width):
                    positions[0].append(position_cursor + time_index)
                    positions[1].append(position_cursor + height_index)
                    positions[2].append(position_cursor + width_index)

        position_cursor += max(time, merged_height, merged_width)
        token_cursor = image_end

    if image_token_id in tokens[token_cursor:]:
        raise ValueError("input contains an image token without a vision grid")
    _append_text_positions(
        positions, position_cursor, len(tokens) - token_cursor)
    if any(len(axis) != len(tokens) for axis in positions):
        raise ValueError("M-RoPE positions do not match the input token count")
    return positions
