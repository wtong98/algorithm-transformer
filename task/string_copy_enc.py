"""
Copying task based on duplicate string task from https://arxiv.org/abs/2305.16843
"""

# <codecell>
import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

start_char = 97    # ASCII 97 corresponds to 'a'

try:
    from .string_copy import CopyDataset
except ImportError:
    from string_copy import CopyDataset


class CopyEncDataset(CopyDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        return self

    def __next__(self):
        rng = np.random.default_rng(self.seed)
        vocab_idxs = [self.tok_to_idx[tok] for tok in self.vocab_toks]
        length = rng.choice(self.lengths, p=self.probs)

        pattern = rng.choice(vocab_idxs, size=length)
        pattern_mask = np.ones(len(pattern))

        xs = np.concatenate((pattern, np.ones(length)))   # GO == 1
        ys = np.concatenate((pattern, pattern))
        pred_mask = np.concatenate((pattern_mask, pattern_mask))

        return xs, ys, pred_mask


def pad_examples(exs):
    xs, ys, masks = zip(*exs)
    max_len = np.max([len(x) for x in xs])

    xs_pad, ys_pad, mask_pad = np.zeros((3, len(exs), max_len))
    for i, (x, y, m) in enumerate(zip(xs, ys, masks)):
        xs_pad[i,:len(x)] = x
        ys_pad[i,:len(y)] = y
        mask_pad[i,:len(m)] = m
    
    return {
        'inputs': jnp.array(xs_pad).astype('int32'),
        'outputs': jnp.array(ys_pad).astype('int32'),
        'mask': jnp.array(mask_pad)
    }


def to_dataloader(ds, batch_size=32, **kwargs):
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=pad_examples, **kwargs)
    return dl

if __name__ == '__main__':
    ds = CopyEncDataset([1,3], weight_prop=True, max_item_label=-1)
    dl = to_dataloader(ds, batch_size=8)
    xs, ys, pad = next(iter(ds))
    # ex = ex[:len(ex)//2]
    # print(xs)
    # print(ys)
    # print(pad)
    print(next(iter(dl)))
# %%
