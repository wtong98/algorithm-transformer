"""
Some nifty utils

author: William Tong (wtong@g.harvard.edu)
"""

from typing import Callable, Optional

from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict

import numpy as np


def new_seed(): return np.random.randint(1, np.iinfo(np.int32).max)


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int | None = None
    num_layers: int = 6
    emb_dim: int = 128
    mlp_dim: int = 128
    max_len: int = 256
    causal: bool = True
    include_bos: bool = True
    ds_generator_name: str = 'RandomGenerator'
    ds_generator_kwargs: FrozenDict = struct.field(default_factory=FrozenDict)
    kernel_init_name: str = 'xavier_uniform'
    kernel_init_params: FrozenDict = struct.field(default_factory=FrozenDict)
    bias_init_name: str = 'normal'
    bias_init_params: FrozenDict = struct.field(default_factory=lambda: FrozenDict({'stddev': 1e-6}))
    posemb_init: Optional[Callable] = None
    posemb_scramble: bool = False
    max_item_label: int = -1
    freeze_embedding: bool = False
    sinus_embedding: bool = False
    nope_embeding: bool = False
    rel_pos_att: bool = False
    rel_pos_rand_max: int = 0

    def kernel_init(self):
        init_f = getattr(nn.initializers, self.kernel_init_name)
        return init_f(**self.kernel_init_params)
    
    def bias_init(self):
        init_f = getattr(nn.initializers, self.bias_init_name)
        return init_f(**self.bias_init_params)