"""
A simple copying task
"""

# <codecell>
import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

start_char = 97    # ASCII 97 corresponds to 'a'

class CopyDataset(IterableDataset):
    def __init__(self, length, vocab_size) -> None:
        self.vocab_size = vocab_size
        self.length = length

        self.vocab_toks = [chr(97 + i) for i in range(vocab_size)]
        self.idx_to_tok = [
            'PAD',
            'GO',
            'END',
        ] + self.vocab_toks
        self.tok_to_idx = {val:i for i, val in enumerate(self.idx_to_tok)}

        self.seed = None
        worker_info = get_worker_info()
        if worker_info != None:
            self.seed = get_worker_info().seed  # TODO: confirm is different
    
    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.seed)
        vocab_idxs = [self.tok_to_idx[tok] for tok in self.vocab_toks]

        pattern = rng.choice(vocab_idxs, size=self.length)
        n_compl = rng.integers(0, len(pattern), endpoint=True)
        xs = np.concatenate((pattern, [self.tok_to_idx['GO']], pattern[:n_compl]))
        if n_compl == len(pattern):
            y = self.tok_to_idx['END']
        else:
            y = pattern[n_compl]
        return xs, y


def pack_examples(exs, max_len, batch_size, pad_tok):
    max_pack_len = max_len * batch_size
    xs, ys = zip(*exs)

    xs_len = [len(x) for x in xs]
    n_final_pad = max_pack_len - np.sum(xs_len)
    final_pad = [[pad_tok] * n_final_pad]
    xs_pack = np.concatenate(list(xs) + final_pad)

    xs_seg = [[i+1] * x_len for i, x_len in enumerate(xs_len)]
    xs_seg = np.concatenate(xs_seg + final_pad)

    xs_pos = [np.arange(x_len) for x_len in xs_len]
    xs_pos = np.concatenate(xs_pos + final_pad)

    return {
        'inputs': xs_pack,
        'inputs_segmentation': xs_seg,
        'inputs_position': xs_pos,
        'targets': ys
    }

def pad_examples(exs, pad_tok):
    xs, ys = zip(*exs)
    xs_len = [len(x) for x in xs]
    max_len = np.max(xs_len)

    out = np.ones((len(exs), max_len)) * pad_tok
    for i, x in enumerate(xs):
        out[i,:len(x)] = x
    
    return {
        'inputs': jnp.array(out),
        'targets': jnp.array(ys)
    }

# ds = CopyDataset(3,2)
# exs = [next(ds) for _ in range(3)]
# pad_examples(exs, 0)

def to_dataloader(ds, batch_size=32, **kwargs):
    # def collate_fn(exs): return pack_examples(exs, 
    #     max_len=max_len, 
    #     batch_size=batch_size, 
    #     pad_tok=ds.tok_to_idx['PAD'])
    def collate_fn(exs): return pad_examples(exs, 
        pad_tok=ds.tok_to_idx['PAD'])
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
    return dl

ds = CopyDataset(3, 2)
dl = to_dataloader(ds, batch_size=5)
next(iter(dl))