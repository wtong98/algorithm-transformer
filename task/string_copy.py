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
from util import TransformerConfig, new_seed

start_char = 97    # ASCII 97 corresponds to 'a'

class BaseGenerator:
    def __init__(self):
        self.alphabet_size = None

    def __next__(self):
        raise NotImplementedError('__next__ not implemented for BaseGenerator')
    

class RandomGenerator(BaseGenerator):
    def __init__(self, lengths, alphabet_size=2,
             ordered=False, unique=False,
             probs=None, sampling_strategy='zipf', 
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
    def __init__(self, lengths, t_lengths=3, nt_ordered=True, nt_unique=True,
                 n_nonterminals=5, n_terminals=10,
                 within_overlap_prob=None, cross_overlap_prob=None,
                 sampling_strategy='zipf', compress=True, seed=None) -> None:
        
        if seed == None:
            seed = new_seed()
        print('info: setting seed', seed)

        self.nt_gen = RandomGenerator(lengths, alphabet_size=n_nonterminals, 
                                     ordered=nt_ordered, 
                                     unique=nt_unique, 
                                     sampling_strategy=sampling_strategy, 
                                     seed=None) # NOTE: high-level generation will be random

        self.within_overlap_prob = within_overlap_prob
        self.cross_overlap_prob = cross_overlap_prob
        rng = np.random.default_rng(seed=seed+1)

        
        self.nt_to_ts = {}
        t_gen = RandomGenerator(t_lengths, alphabet_size=n_terminals, seed=seed+2)

        # first pass: within-tuple duplicates
        for nt in range(n_nonterminals):
            proposal = next(t_gen)['pattern']
            make_same_idxs = rng.binomial(n=1, p=self.within_overlap_prob or 0, size=len(proposal)).astype(bool)
            make_same_idxs = make_same_idxs.astype(bool)
            choices = proposal[make_same_idxs]
            if len(choices) > 0:
                val = rng.choice(choices)
                proposal[make_same_idxs] = val

            self.nt_to_ts[nt] = proposal
        
        # second pass: cross-tuple duplicates
        for nt, tup in self.nt_to_ts.items():
            n_choices = rng.binomial(n=len(tup), p=self.cross_overlap_prob or 0)
            if n_choices > 0:
                for idx in rng.choice(len(tup), size=n_choices, replace=True):
                    other_nt = rng.choice([sym for sym in self.nt_to_ts.keys() if sym != nt])
                    other_t = rng.choice(self.nt_to_ts[other_nt])
                    self.nt_to_ts[nt][idx] = other_t

        if compress:
            self.nt_to_ts, n_terminals = CfgGenerator._remap(self.nt_to_ts)
        self.alphabet_size = n_terminals
    
    @staticmethod
    def _remap(nt_to_ts):
        tok_counter = 0
        compressed_map = {}
        
        for nt in nt_to_ts:
            for i, t in enumerate(nt_to_ts[nt]):
                if t not in compressed_map:
                    compressed_map[t] = tok_counter
                    tok_counter += 1

                t = compressed_map[t]
                nt_to_ts[nt][i] = t
                    
        return nt_to_ts, tok_counter
    
    def __next__(self):
        nts = next(self.nt_gen)['pattern']
        ts = [self.nt_to_ts[nt] for nt in nts]
        ts = [t for chunk in ts for t in chunk]

        return {'pattern': np.array(ts)}
    
    def to_emb_idxs(self, start_idx=4):
        return {k : v + start_idx for k, v in self.nt_to_ts.items()}


def from_config(ds_class, config: TransformerConfig, unify_config=True):
    mod = sys.modules[__name__]
    gen_class = getattr(mod, config.ds_generator_name)
    gen = gen_class(**config.ds_generator_kwargs)
    ds = ds_class(gen, bos=config.include_bos, prompt_mask=config.non_causal_prompt)

    if unify_config:
        config = config.replace(vocab_size=ds.n_symbols)
        if hasattr(gen, 'max_item_label'):
            config = config.replace(max_item_label=gen.max_item_label)
        
    return ds, config


class CopyDataset(IterableDataset):
    def __init__(self, generator: BaseGenerator, bos=True, prompt_mask=False) -> None:
        self.gen = generator
        self.bos = bos
        self.prompt_mask = prompt_mask

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
        return from_config(CopyDataset, config, unify_config=unify_config)
    
    def __iter__(self):
        return self

    def __next__(self):
        out = next(self.gen)
        pattern = out.get('pattern') + self.start_symbol_idx
        item_labels = out.get('labels')

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

        if item_labels is not None:
            item_labels = np.concatenate(([0] if self.bos else [], item_labels, [0], item_labels))  # reflect copy operation
        
        pred_prompt_mask = None
        if self.prompt_mask:
            pred_prompt_mask = np.concatenate((
                [1] if self.bos else[],
                pattern_mask,
                [1],
                0 * pattern_mask,
                [0]
            ))

        return xs, item_labels, pred_mask, pred_prompt_mask


class GenerativeDataset(IterableDataset):
    def __init__(self, generator: BaseGenerator, bos: bool = True) -> None:
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
        return from_config(GenerativeDataset, config, unify_config=unify_config)
    
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
        ))

        pred_mask = np.concatenate((
            [1] if self.bos else [],
            1 * pattern_mask[:-1],
            [0]  # ignore final token
        ))

        if item_labels is None:
            item_labels = np.zeros(length)
        item_labels = np.concatenate(([0] if self.bos else [], item_labels))

        return xs, item_labels, pred_mask
        

def pad_examples(exs):
    items = list(zip(*exs))
    max_len = np.max([len(x) for x in items[0]])
    padded = np.zeros((len(items), len(exs), max_len))
    for i, item_set in enumerate(items):
        for j, item in enumerate(item_set):
            if item is not None:
                padded[i,j,:len(item)] = item

    return {
        'inputs': jnp.array(padded[0]).astype('int32'),
        'labels': jnp.array(padded[1]).astype('int32'),
        'mask': jnp.array(padded[2]),
        'prompt_mask': jnp.array(padded[3])
    }


def to_dataloader(ds, batch_size=32, **kwargs):
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=pad_examples, **kwargs)
    return dl

if __name__ == '__main__':
    config = TransformerConfig(
        ds_generator_name='CfgGenerator',
        ds_generator_kwargs={
            'lengths': np.arange(5) + 1,
            't_lengths': 3,
            'n_terminals': 100000,
            'nt_ordered': False,
            'n_nonterminals': 15,
            'seed': np.random.randint(0, 999999),
            'within_overlap_prob': 0,
            'cross_overlap_prob': 1,
            'compress': True,
        },
        non_causal_prompt=True
    )
    # TODO: benchmark <-- STOPPED HERE
    ds, config = CopyDataset.from_config(config)
    
    dl = to_dataloader(ds, batch_size=8)
    # print(next(iter(dl)))
    print(ds.gen.nt_to_ts)
# %%
# import math

# n = 1000
# k = 15 * 3
# 1 - (math.factorial(n) / (n ** k * math.factorial(n - k)))
