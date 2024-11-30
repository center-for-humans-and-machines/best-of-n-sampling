from typing import Callable, Optional

import accelerate
import torch
from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelGenerateOutput
from transformers import GenerationConfig, PreTrainedModel
from ykutil import (
    approx_number_split,
    pad_along_dimension,
)

from best_of_n.util import get_seq_and_val


def _sample_best_of_n(
    model: HeadedModel,
    generate_args: GenerationConfig,
    input_ids: torch.Tensor | list,
    value_func: Optional[HeadedModel] = None,
    value_head_name: str = "value_head",
    n_samples=4,
    batch_size=4,
    **generate_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    model: HeadedModel,
    generate_args: GenerationConfig,
    input_ids: torch.Tensor | list,
    value_func: Optional[HeadedModel] = None,
    value_head_name: str = "value_head",
    accelerator: Optional[accelerate.Accelerator] = None,
    n_samples=16,
    batch_size=4,
    **generate_kwargs,
) -> list[tuple[list[int], float]]:
    """Perform best-of-n sampling with a value function.

    Args:
        model: A transformer_heads model with a language modelling head and optionally a value head.
        generate_args: The generation config to use for sampling.
        input_ids: The context in which a completion should be generated.
        value_func: Optional transformer_heads value model if model does not have a value head.
        value_head_name: The name of the value head in the model.
        accelerator: Needs to be provided in multi-gpu setting.
        n_samples: The number of samples to generate.
        batch_size: The batch size to use for generation.
        generate_kwargs: Additional keyword arguments to pass to model.generate.

    Returns:
        A sorted list of tuples with the generated sequences and their corresponding values
    """
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
