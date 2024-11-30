from typing import Optional

import accelerate
import torch
from transformer_heads.model.model import HeadedModel
from ykutil import (
    approx_list_split,
    pad_along_dimension,
)

from best_of_n.util import get_seq_and_val


@torch.inference_mode()
def _value_rank(
    model: HeadedModel,
    context: torch.Tensor | list,
    completions: torch.Tensor,
    batch_size: int = 4,
    head_name: str = "value_head",
) -> torch.Tensor:
    context = torch.tensor(context).to(model.device)
    if context.dim() == 1:
        context = context.unsqueeze(0)

    completed = torch.cat([context.repeat(len(completions), 1), completions], dim=1)

    all_values = []

    for i in range(0, len(completed), batch_size):
        batch = completed[i : i + batch_size]
        outputs = model(batch)
        values = outputs.preds_by_head[head_name].squeeze(-1)
        all_values.append(values[:, context.size(1) :])

    all_values = torch.cat(all_values)

    return all_values


def value_rank_completions(
    model: HeadedModel,
    context: torch.Tensor | list,
    completions: list[torch.Tensor | list],
    pad_token_id: int,
    batch_size: int = 4,
    head_name: str = "value_head",
    accelerator: Optional[accelerate.Accelerator] = None,
) -> list[tuple[list[int], float]]:
    """Use a value model to rank a list of completions.

    Args:
        model: A transformer_heads model with a value head.
        context: The tokens coming before the completions.
        completions: A list of possible completions to rank.
        pad_token_id: The token id to use for padding.
        batch_size: Query the model in batches of this size.
        head_name: The name of the value head in the model.
        accelerator: Needs to be provided in multi-gpu setting.
    Returns:
        A sorted list of tuples with the completions and its values.
    """
    gpus = max(1, torch.cuda.device_count())
    assert (
        gpus == 1 or accelerator is not None
    ), "need to provide an accelerator in multi-gpu setting"

    completions = [torch.tensor(x).to(model.device) for x in completions]
    completions = pad_along_dimension(completions, dim=0, pad_value=pad_token_id)

    my_completions = (
        completions
        if gpus == 1
        else approx_list_split(completions, gpus)[accelerator.process_index]
    )

    values = _value_rank(
        model=model,
        context=context,
        completions=my_completions,
        batch_size=batch_size,
        head_name=head_name,
    )

    if accelerator is None:
        values = values
    else:
        values = accelerator.pad_across_processes(values, dim=1, pad_index=0)
        values = accelerator.gather(values)

        accelerator.wait_for_everyone()

    return get_seq_and_val(completions, values, pad_token_id)
