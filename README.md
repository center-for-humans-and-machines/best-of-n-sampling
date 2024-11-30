# Best of n sampling (Value Ranking)
Best-of-n sampling, also known as value ranking incorporates a value function into the process of selecting an action or completion from an LLM.
In a given situation, n possible completions are generated and ranked using a value function. Only the completion with highest predicted value
is then actually chosen.

In situations such as game, in which an optimization of LLM generations towards the outcome of the game is desired, best-of-n sampling is commonly chosen
instead of reinforcement learning due to it's much lower implementational and computational footprint.

While there are other implementations out there ([trl](https://github.com/huggingface/trl/blob/148b5923135e6eaa1e1dfd2c53ce45b274ec3127/trl/extras/best_of_n_sampler.py)), this implementation works natively with multiple gpus (using accelerate) and builds on [transformer_heads](git@github.com:center-for-humans-and-machines/transformer-heads.git)

## Installation
1. Clone this repository
2. From repository root `pip install -e .`

## Usage
* **Rank a list of actions/completions:**
    + Requires a transformer_heads model with a value head
    + `from best_of_n import value_rank_completions`
    + Check [tests/test_mock_value_rank.py](tests/test_mock_value_rank.py) for a usage example.
* **Generate and rank completions:**
    + Requires either a transformer_heads model with language modelling and value head or two transformer_heads models with each doing one of the two tasks.
    + `from best_of_n import sample_best_of_n`
    + Check [tests/test_real_model.py](tests/test_real_model.py) for a usage example.
