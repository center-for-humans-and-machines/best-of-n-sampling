from typing import Callable, Optional

import accelerate
import torch
from torch.nn.utils.rnn import pad_sequence
from transformer_heads.model.model import HeadedModel
from transformer_heads.output import HeadedModelGenerateOutput
from transformers import GenerationConfig, PreTrainedModel
from ykutil import log, pad_along_dimension, removesuffixes


@torch.inference_mode()
def value_rank(
    model: HeadedModel,
    context: torch.Tensor | list,
    completions: list[torch.Tensor | list],
    batch_size: int = 4,
):
    pass


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
    for _ in range(n_samples // batch_size):
        outputs = model.generate(
            inputs=input_ids,
            generation_config=generate_args,
            do_sample=True,
            **generate_kwargs,
        )
        if isinstance(outputs, HeadedModelGenerateOutput):
            out_seq = outputs.sequences
        else:
            out_seq = outputs
        sequences.append(out_seq[:, input_ids.size(1) :])
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
                    :, input_ids.size(1) :
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
        n_samples % gpus == 0
    ), f"n_samples must be divisible by the number of GPUs ({gpus}, got {n_samples})"

    assert (
        gpus == 1 or accelerator is not None
    ), "need to provide an accelerator in multi-gpu setting"

    sequences, values = _sample_best_of_n(
        model=model,
        generate_args=generate_args,
        input_ids=input_ids,
        value_func=value_func,
        value_head_name=value_head_name,
        n_samples=n_samples // gpus,
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

    # print(accelerator.process_index, sequences.shape, values.shape)

    seq_and_val = []

    for seq, val in zip(sequences, values):
        seq.detach().cpu()
        short_seq = removesuffixes(seq.tolist(), (pad_token_id,))
        one_val = float(val[len(short_seq) - 1])
        seq_and_val.append((short_seq, one_val))

    seq_and_val.sort(key=lambda x: x[1], reverse=True)
    return seq_and_val
