# Copyright (c) 2026 Segno System.
"""Prompt-token preparation shared by multimodal runtime paths."""

from operator import index


def _embedding_length(value):
    if isinstance(value, bool):
        raise ValueError("embedding lengths must be positive integers")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(
            "embedding lengths must be positive integers") from exc
    if value <= 0:
        raise ValueError("embedding lengths must be positive integers")
    return value


def _target_span_lengths(tokens, target_token_id):
    lengths = []
    current_length = 0
    for token in tokens:
        if token == target_token_id:
            current_length += 1
        elif current_length:
            lengths.append(current_length)
            current_length = 0
    if current_length:
        lengths.append(current_length)
    return lengths


def prepare_multimodal_tokens(
    input_tokens,
    embedding_lengths,
    begin_token_id,
    end_token_id,
    target_token_id,
):
    """Expand empty media markers or validate existing target-token spans."""
    tokens = list(input_tokens)
    expected_lengths = [
        _embedding_length(length) for length in embedding_lengths
    ]

    if target_token_id in tokens:
        span_lengths = _target_span_lengths(tokens, target_token_id)
        if len(span_lengths) != len(expected_lengths):
            raise ValueError(
                "multimodal target spans and embedding results must match")
        for span_index, (actual, expected) in enumerate(
            zip(span_lengths, expected_lengths)
        ):
            if actual != expected:
                raise ValueError(
                    f"multimodal span {span_index} has {actual} tokens; "
                    f"expected {expected}")
        return tokens

    marker_positions = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index] == begin_token_id
        and tokens[index + 1] == end_token_id
    ]
    if len(marker_positions) != len(expected_lengths):
        raise ValueError(
            "empty multimodal markers and embedding results must match")

    expanded = []
    marker_index = 0
    token_index = 0
    while token_index < len(tokens):
        expanded.append(tokens[token_index])
        if (
            marker_index < len(marker_positions)
            and token_index == marker_positions[marker_index]
        ):
            embedding_length = expected_lengths[marker_index]
            expanded.extend([target_token_id] * embedding_length)
            marker_index += 1
        token_index += 1
    return expanded
