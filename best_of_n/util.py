import torch
from ykutil import removesuffixes


def get_seq_and_val(
    sequences: torch.Tensor, values: torch.Tensor, pad_token_id: int
) -> list[tuple[list[int], float]]:
    """Removes all padding from sequences, extracts the value from the last token, and sorts by value.

    Args:
        sequences: Tensor representing many padded sequences of tokens.
        values: Tensor containing values for each token in each sequence.
        pad_token_id: The token id that was used for padding.

    Returns:
        list[tuple[list[int], float]]: A sorted list of tuples containing a sequence and a value.
    """
    seq_and_val = []

    for seq, val in zip(sequences, values):
        seq.detach().cpu()
        short_seq = removesuffixes(seq.tolist(), (pad_token_id,))
        one_val = float(val[len(short_seq) - 1])
        seq_and_val.append((short_seq, one_val))

    seq_and_val.sort(key=lambda x: x[1], reverse=True)

    return seq_and_val
