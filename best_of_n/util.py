from ykutil import removesuffixes


def get_seq_and_val(sequences, values, pad_token_id):
    seq_and_val = []

    for seq, val in zip(sequences, values):
        seq.detach().cpu()
        short_seq = removesuffixes(seq.tolist(), (pad_token_id,))
        one_val = float(val[len(short_seq) - 1])
        seq_and_val.append((short_seq, one_val))

    seq_and_val.sort(key=lambda x: x[1], reverse=True)

    return seq_and_val
