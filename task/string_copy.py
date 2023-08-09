"""
A simple copying task
"""

# <codecell>
import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

start_char = 97    # ASCII 97 corresponds to 'a'

class CopyDataset(IterableDataset):
    def __init__(self, lengths, 
                 probs=None, prob_type=None, 
                 vocab_size=2, max_item_label=-1, bos=True,
                 unique=False, ordered=False, seed=None) -> None:
        self.max_item_label = max_item_label
        self.bos = bos

        self.copy_gen = copy_generator(lengths, alphabet_size=vocab_size,
                                       unique=unique, ordered=ordered,
                                       probs=probs, sampling_strategy=prob_type, seed=seed)
        

        self.vocab_toks = [chr(start_char + i) for i in range(vocab_size)]
        self.idx_to_tok = [
            'PAD',
            'GO',
            'END',
            'START'
        ] + self.vocab_toks
        self.tok_to_idx = {val:i for i, val in enumerate(self.idx_to_tok)}
        self.n_symbols = len(self.idx_to_tok)
        self.start_symbol_idx = self.tok_to_idx[self.vocab_toks[0]]
    
    def __iter__(self):
        return self

    def __next__(self):
        pattern = next(self.copy_gen) + self.start_symbol_idx
        length = len(pattern)

        pattern_mask = np.ones(len(pattern))
        xs = np.concatenate((
            [self.tok_to_idx['START']] if self.bos else [], 
            pattern, 
            [self.tok_to_idx['GO']], 
            pattern, 
            [self.tok_to_idx['END']]
        ))

        pred_mask = np.concatenate((
            [0] if self.bos else [],
            0 * pattern_mask,  # ignore prefix
            [1],               # start tracking at GO
            pattern_mask,      # predict output
            [0]                # ignored final prediction
        ))

        if self.max_item_label > 0:
            item_labels = np.sort(np.random.choice(np.arange(1, self.max_item_label + 1), size=length, replace=False))
        else:
            item_labels = np.zeros(length)
        item_labels = np.concatenate(([0] if self.bos else [], item_labels, [0], item_labels))  # reflect copy operation

        return xs, item_labels, pred_mask


def copy_generator(lengths, alphabet_size=2,
                   ordered=False, unique=False,
                   probs=None, sampling_strategy=None, 
                   seed=None):

    try:
        lengths = list(lengths)
    except TypeError:
        lengths = [lengths]
    
    if unique and max(lengths) > alphabet_size:
        raise ValueError('tokens are supposed to be unique, but maximum length exceeds vocab size')
    
    if probs is None:
        if sampling_strategy == 'zipf':
            weights = 1 / np.array(lengths)
            probs = weights / np.sum(weights)
        elif sampling_strategy == 'inv_zipf':
            weights = np.array(lengths)
            probs = weights / np.sum(weights)
        else:
            raise ValueError(f'sampling strategy unrecognized: {sampling_strategy}')
    
    
    rng = np.random.default_rng(seed)
    while True:
        length = rng.choice(lengths, p=probs)
        pattern = rng.choice(alphabet_size, size=length, replace=not unique)
        if ordered:
            pattern = np.sort(pattern)
        
        yield pattern


class CFG:
    def __init__(self, nt_lengths, ordered=True, n_nt=5, n_t=10, seed=None) -> None:
        pass

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
    import sys
    sys.path.append('../')

    ds = CopyDataset([1,2,3], vocab_size=10, prob_type='zipf', max_item_label=-1, bos=True)
    dl = to_dataloader(ds, batch_size=8)
    ex = next(iter(dl))
    # ex = ex[:len(ex)//2]
    print(ex)
    # print(next(iter(ds)))
# %%
