import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import verifiers as vf

from prime_rl.utils.logger import InterceptHandler

if TYPE_CHECKING:
    from prime_rl.transport import TrainingSample


# TODO: remove once usage is tracked by verifiers
def get_prompt_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of prompt tokens from vf.RolloutOutput. Defined as the
    number of prompt ids from the first trajectory step. If raw tokens are not
    available, falls back to checking the usage of the first response.
    """
    if not output["trajectory"]:
        return 0
    first_step = output["trajectory"][0]
    if first_step["tokens"] is not None:
        return len(first_step["tokens"]["prompt_ids"])
    first_step_response = first_step["response"]
    return (first_step_response.get("usage") or {}).get("prompt_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_submitted_seq_len(output: vf.RolloutOutput) -> int:
    """
    Computes the final inference request length, including its completion.

    This is not the logical rollout length when earlier tokens were compacted
    out of a later request.
    """
    if not output["trajectory"]:
        return 0
    last_step = output["trajectory"][-1]
    if last_step["tokens"] is not None:
        return len(last_step["tokens"]["prompt_ids"]) + len(last_step["tokens"]["completion_ids"])
    last_step_response = last_step["response"]
    return (last_step_response.get("usage") or {}).get("total_tokens", 0)


def get_seq_len(output: vf.RolloutOutput) -> int:
    """Gets logical rollout length, falling back to final submitted length."""
    logical_seq_len = get_logical_seq_len(output)
    if logical_seq_len is not None:
        return logical_seq_len
    return get_submitted_seq_len(output)


def get_logical_seq_len(output: vf.RolloutOutput) -> int | None:
    """Gets the monotonic Phase4 rollout length when explicitly recorded."""
    if not output["trajectory"]:
        return 0
    extras = output["trajectory"][-1].get("extras") or {}
    logical_seq_len = extras.get("logical_seq_len")
    if type(logical_seq_len) is not int or logical_seq_len < 0:
        return None
    return logical_seq_len


def _get_last_step_length_extra(
    output: vf.RolloutOutput,
    field: str,
) -> int | None:
    if not output["trajectory"]:
        return 0
    extras = output["trajectory"][-1].get("extras") or {}
    value = extras.get(field)
    if type(value) is not int or value < 0:
        return None
    return value


def get_logical_padding_seq_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "logical_padding_seq_len")


def get_logical_non_padding_seq_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "logical_non_padding_seq_len")


def get_logical_sequence_limit_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "logical_sequence_limit_len")


def get_context_seq_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "context_seq_len")


def get_context_padding_seq_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "context_padding_seq_len")


def get_context_non_padding_seq_len(output: vf.RolloutOutput) -> int | None:
    return _get_last_step_length_extra(output, "context_non_padding_seq_len")


# TODO: remove once usage is tracked by verifiers
def get_completion_len(output: vf.RolloutOutput) -> int:
    """Computes cumulative completion tokens across every trajectory step."""
    completion_len = 0
    for step in output["trajectory"]:
        if step["tokens"] is not None:
            completion_len += len(step["tokens"]["completion_ids"])
        else:
            completion_len += (step["response"].get("usage") or {}).get("completion_tokens", 0)
    return completion_len


def get_training_token_counts(samples: Sequence["TrainingSample"]) -> tuple[int, int]:
    """Returns aggregate prefill and decode tokens in untruncated training samples."""
    total_tokens = sum(len(sample.prompt_ids) + len(sample.completion_ids) for sample in samples)
    decode_tokens = sum(sum(sample.completion_mask) for sample in samples)
    return total_tokens - decode_tokens, decode_tokens


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
