import fire
from accelerate import Accelerator
from mock_model import MockModel
from transformers import AutoTokenizer, GenerationConfig

from best_of_n import sample_best_of_n


def test_without_accelerator(device="cpu"):
    tk = AutoTokenizer.from_pretrained("gpt2")
    tk.pad_token_id = tk.eos_token_id
    model = MockModel(tk, device, True)
    sampled = sample_best_of_n(
        model,
        GenerationConfig(pad_token_id=tk.pad_token_id),
        [tk.encode("A car is")],
        n_samples=12,
    )
    sampled = [(tk.decode(x), v) for x, v in sampled]
    assert len(sampled) == 12

    model_not_value = MockModel(tk, device, False)
    sampled = sample_best_of_n(
        model_not_value,
        GenerationConfig(pad_token_id=tk.pad_token_id),
        [tk.encode("A car is")],
        n_samples=12,
        value_func=model,
    )
    sampled = [(tk.decode(x), v) for x, v in sampled]
    assert len(sampled) == 12


def test_with_accelerator():
    accerlator = Accelerator()
    tk = AutoTokenizer.from_pretrained("gpt2")
    tk.pad_token_id = tk.eos_token_id
    model = MockModel(tk, accerlator.device, True)
    text = "Player 1 (liberal): "
    tok_text = tk.encode(text, return_tensors="pt")

    gen_args = GenerationConfig(
        temperature=1.0,
        do_sample=True,
        pad_token_id=tk.pad_token_id,
        max_new_tokens=20,
    )

    accerlator.wait_for_everyone()

    sampled = sample_best_of_n(
        model,
        accelerator=accerlator,
        generate_args=gen_args,
        input_ids=tok_text,
        n_samples=16,
    )
    sampled = [(tk.decode(x), v) for x, v in sampled]
    assert len(sampled) == 16

    model_not_value = MockModel(tk, accerlator.device, False)
    sampled = sample_best_of_n(
        model_not_value,
        accelerator=accerlator,
        generate_args=gen_args,
        input_ids=tok_text,
        n_samples=16,
        value_func=model,
    )
    assert len(sampled) == 16


if __name__ == "__main__":
    fire.Fire()
