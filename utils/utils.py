import torch


def get_padding_mask(wavs, durations):

    max_len = wavs.size(-1)

    out = [
        [1] * int(max_len * ln) + [0] * (max_len - int(max_len * ln))
        for ln in durations
    ]

    return torch.LongTensor(out)
