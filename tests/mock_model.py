import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformer_heads.output import HeadedModelGenerateOutput, HeadedModelOutput
from transformers import PreTrainedTokenizer


class MockModel:
    def __init__(self, tk: PreTrainedTokenizer, device: str, is_value_model=True):
        self.tk = tk
        self.response_list = [
            tk.encode(x, return_tensors="pt")
            for x in [
                "is great",
                "is very nice",
                "some beautiful nieche thing",
                "because cheese.\n",
                "blablablabla",
                "short",
                "what is this very long thing here?",
            ]
        ]
        self.device = device
        self.config = None
        self.is_value_model = is_value_model

    def __call__(self, input_ids: torch.Tensor):
        return HeadedModelOutput(
            preds_by_head={"value_head": torch.rand(input_ids.size())}
        )

    def generate(self, inputs: torch.Tensor, *_args, **_kwargs):
        random.shuffle(self.response_list)
        out_pred = [
            torch.cat(
                [
                    inputs[i],
                    self.response_list[i % len(self.response_list)][0].to(
                        inputs.device
                    ),
                ]
            )
            for i in range(len(inputs))
        ]
        out_seq = pad_sequence(
            out_pred, batch_first=True, padding_value=self.tk.pad_token_id
        )
        if not self.is_value_model:
            return out_seq
        out_vals = torch.rand(out_seq.size())

        return HeadedModelGenerateOutput(
            sequences=out_seq,
            head_outputs={"value_head": out_vals},
        )
