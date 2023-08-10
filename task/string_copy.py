"""
A simple copying task
"""

# <codecell>
import sys

import jax.numpy as jnp
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import sys
sys.path.append('../')
from util import TransformerConfig

start_char = 97    # ASCII 97 corresponds to 'a'

class BaseGenerator:
    def __init__(self):
        self.alphabet_size = None

    def __next__(self):
        raise NotImplementedError('__next__ not implemented for BaseGenerator')
    

class RandomGenerator(BaseGenerator):
    def __init__(self, lengths, alphabet_size=2,
             ordered=False, unique=False,
             probs=None, sampling_strategy=None, 
             seed=None):

        try:
            lengths = list(lengths)
        except TypeError:
            lengths = [lengths]
        
        if unique and max(lengths) > alphabet_size:
            raise ValueError('tokens are supposed to be unique, but maximum length exceeds vocab size')
        
        if probs is None and sampling_strategy is not None:
            if sampling_strategy == 'zipf':
                weights = 1 / np.array(lengths)
                probs = weights / np.sum(weights)
            elif sampling_strategy == 'inv_zipf':
                weights = np.array(lengths)
                probs = weights / np.sum(weights)
            else:
                raise ValueError(f'sampling strategy unrecognized: {sampling_strategy}')
        
        self.lengths = lengths
        self.ordered = ordered
        self.unique = unique
        self.probs = probs
        self.rng = np.random.default_rng(seed)
        self.alphabet_size = alphabet_size
    
    def __next__(self):
        length = self.rng.choice(self.lengths, p=self.probs)
        pattern = self.rng.choice(self.alphabet_size, size=length, replace=not self.unique)
        if self.ordered:
            pattern = np.sort(pattern)
        
        return {'pattern': pattern}


class RandomGeneratorWithLabels(RandomGenerator):
    def __init__(self, lengths, max_item_label, **kwargs) -> None:
        super().__init__(lengths, **kwargs)
        self.max_item_label = max_item_label
    
    def __next__(self):
        pattern = super().__next__()['pattern']
        length = len(pattern)

        labels = np.random.choice(np.arange(1, self.max_item_label + 1), size=length, replace=False)
        labels = np.sort(labels)
        return {'pattern': pattern, 'labels': labels}


class CfgGenerator(BaseGenerator):
    def __init__(self, nt_lengths, t_lengths=2, nt_ordered=True, 
                 n_nonterminals=5, n_terminals=10, 
                 sampling_strategy='zipf', seed=0) -> None:
        self.nt_gen = RandomGenerator(nt_lengths, alphabet_size=n_nonterminals, 
                                     ordered=nt_ordered, 
                                     unique=nt_ordered, 
                                     sampling_strategy=sampling_strategy, 
                                     seed=seed)

        t_gen = RandomGenerator(t_lengths, alphabet_size=n_terminals, seed=seed+1)
        self.nt_to_ts = {nt: next(t_gen)['pattern'] for nt in range(n_nonterminals)}
        self.alphabet_size = n_terminals
    
    def __next__(self):
        nts = next(self.nt_gen)['pattern']
        ts = [self.nt_to_ts[nt] for nt in nts]
        ts = [t for chunk in ts for t in chunk]

        return {'pattern': np.array(ts)}
    
    def to_emb_idxs(self, start_idx):
        return {k : v + start_idx for k, v in self.nt_to_ts.items()}


class CopyDataset(IterableDataset):
    def __init__(self, generator: BaseGenerator, bos=True) -> None:
        self.gen = generator
        self.bos = bos

        self.vocab_toks = [chr(start_char + i) for i in range(self.gen.alphabet_size)]
        self.idx_to_tok = [
            'PAD',
            'GO',
            'END',
            'START'
        ] + self.vocab_toks
        self.tok_to_idx = {val:i for i, val in enumerate(self.idx_to_tok)}
        self.n_symbols = len(self.idx_to_tok)
        self.start_symbol_idx = self.tok_to_idx[self.vocab_toks[0]]
    
    @staticmethod
    def from_config(config: TransformerConfig, unify_config=True):
        mod = sys.modules[__name__]
        gen_class = getattr(mod, config.ds_generator_name)
        gen = gen_class(**config.ds_generator_kwargs)
        ds = CopyDataset(gen, bos=config.include_bos)

        if unify_config:
            config = config.replace(vocab_size=ds.n_symbols)
            if hasattr(gen, 'max_item_label'):
                config = config.replace(max_item_label=gen.max_item_label)
            
        return ds, config
    
    def __iter__(self):
        return self

    def __next__(self):
        out = next(self.gen)
        pattern = out.get('pattern') + self.start_symbol_idx
        item_labels = out.get('labels')
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

        if item_labels is None:
            item_labels = np.zeros(length)
        item_labels = np.concatenate(([0] if self.bos else [], item_labels, [0], item_labels))  # reflect copy operation

        return xs, item_labels, pred_mask


def pad_examples(exs):
    xs, item_labels, masks = zip(*exs)
    max_len = np.max([len(x) for x in xs])

    xs_pad = np.zeros((len(exs), max_len))
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
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=pad_examples, **kwargs)
    return dl

if __name__ == '__main__':
    config = TransformerConfig(
        ds_generator_name='CfgGenerator',
        ds_generator_kwargs={
            'nt_lengths': np.arange(5) + 1,
        }
    )

    ds, config = CopyDataset.from_config(config)
    
    dl = to_dataloader(ds, batch_size=8)
    print(next(iter(dl)))
# %%
