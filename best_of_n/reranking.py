import torch
from scipy.special import log_softmax
from transformers import PreTrainedModel
from ykutil import compute_seq_log_probability


def rerank_seq_and_val(
    model: PreTrainedModel,
    inputs: list[int] | torch.Tensor,
    seq_and_val: list[tuple[list[int], float]],
    value_temperature=100,
) -> list[tuple[list[int], float]]:
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.squeeze().tolist()
    value_logprobs = log_softmax([x[1] * value_temperature for x in seq_and_val])

    # Batched would be more efficient
    seq_probs = log_softmax(
        [
            compute_seq_log_probability(
                model=model, pre_seq_tokens=inputs, post_seq_tokens=x[0]
            )
            for x in seq_and_val
        ]
    )

    logprobs = value_logprobs + seq_probs
    seq_and_val = [(x[0], lp) for lp, x in zip(logprobs, seq_and_val)]

    seq_and_val.sort(key=lambda x: -x[1])

    return seq_and_val
