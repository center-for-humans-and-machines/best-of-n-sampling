import fire
from accelerate import Accelerator
from mock_model import MockModel
from transformers import AutoTokenizer, GenerationConfig

from best_of_n import value_rank


def test_without_accelerator(device="cpu"):
    tk = AutoTokenizer.from_pretrained("gpt2")
    tk.pad_token_id = tk.eos_token_id
    model = MockModel(tk, device, True)
    inputs = tk.encode("A car is ", return_tensors="pt").to(model.device)
    completions = model.generate(inputs.repeat(12, 1)).sequences[:, inputs.size(1) :]

    seq_and_val = value_rank(
        model,
        inputs,
        completions,
        tk.pad_token_id,
    )

    seq_and_val = [(tk.decode(x), v) for x, v in seq_and_val]
    assert len(seq_and_val) == 12

    print(seq_and_val)


if __name__ == "__main__":
    fire.Fire()
