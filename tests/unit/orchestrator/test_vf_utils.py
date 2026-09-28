from types import SimpleNamespace

from prime_rl.orchestrator.vf_utils import (
    get_completion_len,
    get_context_non_padding_seq_len,
    get_context_padding_seq_len,
    get_context_seq_len,
    get_logical_non_padding_seq_len,
    get_logical_padding_seq_len,
    get_logical_seq_len,
    get_logical_sequence_limit_len,
    get_seq_len,
    get_submitted_seq_len,
    get_training_token_counts,
)


def _sample(prompt_len: int, completion_mask: list[bool]) -> SimpleNamespace:
    completion_len = len(completion_mask)
    return SimpleNamespace(
        prompt_ids=list(range(prompt_len)),
        completion_ids=list(range(completion_len)),
        completion_mask=completion_mask,
    )


def test_completion_len_sums_every_trajectory_step():
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4],
                }
            },
            {
                "tokens": {
                    "prompt_ids": [1, 4, 5],
                    "completion_ids": [6, 7, 8],
                }
            },
        ]
    }

    assert get_completion_len(output) == 5
    assert get_submitted_seq_len(output) == 6
    assert get_seq_len(output) == 6
    assert get_logical_seq_len(output) is None


def test_seq_len_prefers_explicit_logical_length():
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 4, 5],
                    "completion_ids": [6, 7, 8],
                },
                "extras": {"logical_seq_len": 17},
            }
        ]
    }

    assert get_logical_seq_len(output) == 17
    assert get_seq_len(output) == 17


def test_padding_and_context_lengths_are_read_from_final_step_extras():
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3],
                },
                "extras": {
                    "logical_padding_seq_len": 7,
                    "logical_non_padding_seq_len": 13,
                    "logical_sequence_limit_len": 13,
                    "context_seq_len": 12,
                    "context_padding_seq_len": 4,
                    "context_non_padding_seq_len": 8,
                },
            }
        ]
    }

    assert get_logical_padding_seq_len(output) == 7
    assert get_logical_non_padding_seq_len(output) == 13
    assert get_logical_sequence_limit_len(output) == 13
    assert get_context_seq_len(output) == 12
    assert get_context_padding_seq_len(output) == 4
    assert get_context_non_padding_seq_len(output) == 8


def test_completion_len_supports_usage_fallback_for_each_step():
    output = {
        "trajectory": [
            {
                "tokens": None,
                "response": {
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    }
                },
            },
            {
                "tokens": None,
                "response": {
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 5,
                        "total_tokens": 9,
                    }
                },
            },
        ]
    }

    assert get_completion_len(output) == 8
    assert get_submitted_seq_len(output) == 9


def test_full_context_training_length_matches_final_submitted_sequence():
    samples = [_sample(prompt_len=2, completion_mask=[True, True, False, False, True, True])]
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4],
                }
            },
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3, 4, 5, 6],
                    "completion_ids": [7, 8],
                }
            }
        ]
    }

    prefill_tokens, decode_tokens = get_training_token_counts(samples)

    assert prefill_tokens == 4
    assert decode_tokens == 4
    assert prefill_tokens + decode_tokens == get_submitted_seq_len(output)
    assert get_completion_len(output) == 4


def test_compacted_lengths_keep_logical_submitted_and_training_counts_distinct():
    samples = [
        _sample(prompt_len=3, completion_mask=[True, True]),
        _sample(prompt_len=2, completion_mask=[False, True, True, True]),
    ]
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                }
            },
            {
                "tokens": {
                    "prompt_ids": [10, 11],
                    "completion_ids": [12, 13, 14, 15],
                },
                "extras": {"logical_seq_len": 9},
            },
        ]
    }

    prefill_tokens, decode_tokens = get_training_token_counts(samples)

    assert prefill_tokens == 6
    assert decode_tokens == 5
    assert prefill_tokens + decode_tokens == 11
    assert get_submitted_seq_len(output) == 6
    assert get_seq_len(output) == 9
    assert prefill_tokens + decode_tokens > get_seq_len(output)
