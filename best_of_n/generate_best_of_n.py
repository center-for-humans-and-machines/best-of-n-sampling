from typing import Callable, Optional

import accelerate
import torch
from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelGenerateOutput
from transformers import GenerationConfig, PreTrainedModel
from ykutil import (
    approx_list_split,
    approx_number_split,
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
):
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


def value_rank(
    model: HeadedModel,
    context: torch.Tensor | list,
    completions: list[torch.Tensor | list],
    pad_token_id: int,
    batch_size: int = 4,
    head_name: str = "value_head",
    accelerator: Optional[accelerate.Accelerator] = None,
):
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


def _sample_best_of_n(
    model: PreTrainedModel | HeadedModel,
    generate_args: GenerationConfig,
    input_ids: torch.Tensor | list,
    value_func: Optional[HeadedModel] = None,
    value_head_name: str = "value_head",
    n_samples=4,
    batch_size=4,
    **generate_kwargs,
):
    pad_token_id = (
        model.config.pad_token_id
        if hasattr(model.config, "pad_token_id")
        and model.config.pad_token_id is not None
        else generate_args.pad_token_id
    )
    sequences = []
    values = []
    input_ids = torch.tensor(input_ids).to(model.device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.repeat(batch_size, 1)
    for i in range(0, n_samples, batch_size):
        if n_samples - i < batch_size:
            use_input_ids = input_ids[: n_samples - i]
        else:
            use_input_ids = input_ids
        outputs = model.generate(
            inputs=use_input_ids,
            generation_config=generate_args,
            do_sample=True,
            **generate_kwargs,
        )
        if isinstance(outputs, HeadedModelGenerateOutput):
            out_seq = outputs.sequences
        else:
            out_seq = outputs
        sequences.append(out_seq[:, use_input_ids.size(1) :])
        if (
            isinstance(outputs, HeadedModelGenerateOutput)
            and value_head_name in outputs.head_outputs
        ):
            values.append(outputs.head_outputs[value_head_name].squeeze(-1))
        else:
            assert value_func is not None
            value_out = value_func(out_seq)
            values.append(
                value_out.preds_by_head[value_head_name][
                    :, use_input_ids.size(1) :
                ].squeeze(-1)
            )

    sequences = pad_along_dimension(sequences, dim=1, pad_value=pad_token_id)
    values = pad_along_dimension(values, dim=1, pad_value=0.0)
    return sequences, values


def sample_best_of_n(
    model: PreTrainedModel | HeadedModel,
    generate_args: GenerationConfig,
    input_ids: torch.Tensor | list,
    value_func: Optional[Callable] = None,
    value_head_name: str = "value_head",
    accelerator: Optional[accelerate.Accelerator] = None,
    n_samples=16,
    batch_size=4,
    **generate_kwargs,
):
    pad_token_id = (
        model.config.pad_token_id
        if hasattr(model.config, "pad_token_id")
        and model.config.pad_token_id is not None
        else generate_args.pad_token_id
    )
    assert pad_token_id is not None, "need to specify pad_token_id in generation config"
    gpus = max(1, torch.cuda.device_count())
    assert (
        gpus == 1 or accelerator is not None
    ), "need to provide an accelerator in multi-gpu setting"

    my_samples = (
        n_samples
        if gpus == 1
        else approx_number_split(n_samples, gpus)[accelerator.process_index]
    )

    sequences, values = _sample_best_of_n(
        model=model,
        generate_args=generate_args,
        input_ids=input_ids,
        value_func=value_func,
        value_head_name=value_head_name,
        n_samples=my_samples,
        batch_size=batch_size,
        **generate_kwargs,
    )

    if accelerator is None:
        sequences = sequences.flatten(0, 1)
        values = values.flatten(0, 1)
    else:
        sequences = accelerator.pad_across_processes(
            sequences, dim=2, pad_index=pad_token_id
        )
        values = accelerator.pad_across_processes(values, dim=2, pad_index=0)

        sequences = accelerator.gather(sequences).flatten(0, 1)
        values = accelerator.gather(values).flatten(0, 1)

        accelerator.wait_for_everyone()

    return get_seq_and_val(sequences, values, pad_token_id)
