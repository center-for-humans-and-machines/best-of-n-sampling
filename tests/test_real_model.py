import time

import fire
from accelerate import Accelerator
from transformer_heads import load_lora_with_heads, load_tokenizer
from transformers import (
    BitsAndBytesConfig,
    GenerationConfig,
    MistralForCausalLM,
    PreTrainedTokenizer,
)

from best_of_n import sample_best_of_n


def get_prefix_allowed_fn(stop_tk, tokenizer: PreTrainedTokenizer):
    all_tk = list(range(len(tokenizer)))

    def prefix_allowed_fn(_batch_idx, input_ids):
        if len(input_ids) > 0 and input_ids[-1] == stop_tk:
            return [tokenizer.eos_token_id]
        return all_tk

    return prefix_allowed_fn


def test_single_gpu(model_path):
    model = load_lora_with_heads(
        MistralForCausalLM,
        model_path,
        device_map={"": "cuda:0"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )
    tk = load_tokenizer(model_path)
    text = "Player 1 (liberal): "
    tok_text = tk.encode(text, return_tensors="pt")

    gen_args = GenerationConfig(
        temperature=1.0,
        do_sample=True,
        pad_token_id=tk.pad_token_id,
        max_new_tokens=20,
    )

    start = time.perf_counter()

    res = sample_best_of_n(
        model,
        generate_args=gen_args,
        input_ids=tok_text,
        n_samples=16,
        prefix_allowed_tokens_fn=get_prefix_allowed_fn(
            tk.encode("\n", add_special_tokens=False)[-1], tk
        ),
    )
    res = [(tk.decode(x), v) for x, v in res]
    print(res)

    print(model.device)
    print(f"Time taken: {time.perf_counter() - start}s")


def test_multi_gpu(model_path):
    accerlator = Accelerator()
    model = load_lora_with_heads(
        MistralForCausalLM,
        model_path,
        device_map=accerlator.device,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )
    tk = load_tokenizer(model_path)
    text = "Player 1 (liberal): "
    tok_text = tk.encode(text, return_tensors="pt")

    gen_args = GenerationConfig(
        temperature=1.0,
        do_sample=True,
        pad_token_id=tk.pad_token_id,
        max_new_tokens=20,
    )

    accerlator.wait_for_everyone()
    start = time.perf_counter()

    res = sample_best_of_n(
        model,
        accelerator=accerlator,
        generate_args=gen_args,
        input_ids=tok_text,
        n_samples=16,
        prefix_allowed_tokens_fn=get_prefix_allowed_fn(
            tk.encode("\n", add_special_tokens=False)[-1], tk
        ),
    )
    res = [(tk.decode(x), v) for x, v in res]
    if accerlator.is_main_process:
        print(res)

    print(model.device)
    print(f"Time taken: {time.perf_counter() - start}s")


if __name__ == "__main__":
    fire.Fire()
