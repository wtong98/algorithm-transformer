"""
A simple copying task
"""

# <codecell>
import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

start_char = 97    # ASCII 97 corresponds to 'a'


# TODO: add randomized label-item embed here <-- STOPPED HERE
class CopyDataset(IterableDataset):
    def __init__(self, lengths, probs=None, vocab_size=2, weight_prop=False, max_item_label=-1) -> None:
        self.vocab_size = vocab_size
        self.weight_prop = weight_prop
        self.max_item_label = max_item_label

        try:
            self.lengths = list(lengths)
        except TypeError:
            self.lengths = [lengths]

        self.probs = probs
        if self.probs == None and self.weight_prop:
            weights = np.array(self.lengths)
            self.probs = weights / np.sum(weights)

        self.vocab_toks = [chr(start_char + i) for i in range(vocab_size)]
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
        length = rng.choice(self.lengths, p=self.probs)

        pattern = rng.choice(vocab_idxs, size=length)
        pattern_mask = np.ones(len(pattern))
        xs = np.concatenate((pattern, [self.tok_to_idx['GO']], pattern, [self.tok_to_idx['END']]))
        pred_mask = np.concatenate((
            0 * pattern_mask,  # ignore prefix
            [1],               # start tracking at GO
            pattern_mask,      # predict output
            [0]                # ignored final prediction
        ))

        if self.max_item_label > 0:
            item_labels = np.sort(rng.choice(np.arange(1, self.max_item_label + 1), size=length, replace=False))
            item_labels = np.concatenate((item_labels, [0], item_labels))  # reflect copy operation
        else:
            item_labels = np.zeros(length)

        return xs, item_labels, pred_mask


def pad_examples(exs, pad_tok):
    xs, item_labels, masks = zip(*exs)
    max_len = np.max([len(x) for x in xs])

    xs_pad = np.ones((len(exs), max_len)) * pad_tok
    labels_pad = np.zeros((len(exs), max_len))
    mask_pad = np.zeros((len(exs), max_len))
    for i, (x, l, m) in enumerate(zip(xs, item_labels, masks)):
        xs_pad[i,:len(x)] = x
        labels_pad[i,:len(l)] = l
        mask_pad[i,:len(m)] = m
    
    return {
        'inputs': jnp.array(xs_pad).astype('int32'),
        'labels': jnp.array(labels_pad).astype('int32'),
        'mask': jnp.array(mask_pad)
    }


def to_dataloader(ds, batch_size=32, **kwargs):
    def collate_fn(exs): return pad_examples(exs, 
        pad_tok=ds.tok_to_idx['PAD'])
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
    return dl

if __name__ == '__main__':
    ds = CopyDataset([1,3], weight_prop=True, max_item_label=-1)
    dl = to_dataloader(ds, batch_size=8)
    ex = next(iter(ds))[0]
    # ex = ex[:len(ex)//2]
    print(type(ex))
    # print(next(iter(ds)))