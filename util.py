"""
Some nifty utils

author: William Tong (wtong@g.harvard.edu)
"""

from typing import Callable, Optional

from flax import linen as nn, struct
from flax.core.frozen_dict import FrozenDict
from flax.training.common_utils import stack_forest
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

import matplotlib.pyplot as plt
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
    non_causal_prompt: bool = False
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


def plot_train_metrics(info, save_path=None):
    train = stack_forest(info['train_metrics'])
    # test = stack_forest(info['eval_metrics'])

    fig, axs = plt.subplots(1, 1, figsize=(6, 3))

    for ax, metrics in zip([axs], [train]):
        ax.plot(metrics['accuracy'], color='C0', label='accuracy', alpha=0.8)
        ax.set_ylabel('Accuracy', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')

        ax.plot(metrics['aon_accuracy'], color='C0', label='aon_accuracy', alpha=0.6, linestyle='dashed')
        ax.set_xscale('log')

        ax2 = ax.twinx()
        ax2.plot(metrics['loss'], color='C1', label='loss', alpha=0.8)
        ax2.set_ylabel('Loss', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax.set_title('Train metrics')

        # ax.plot(metrics['confidence'], label='confidence')
        # ax.plot(metrics['loss'], label='loss')

    fig.tight_layout()
    fig.legend()

    if save_path is not None:
        plt.savefig(save_path)


def make_ckpt_manager(save_dir):
    return CheckpointManager(
        save_dir, 
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(
                keep_period=1,
                best_fn=lambda x: x,
                best_mode='min')
        
    )


def load_params(save_dir, step=None):
    mngr = make_ckpt_manager(save_dir)
    if step is None:
        step = mngr.best_step()

    r = mngr.restore(step)
    raw_state = r['state']
    params = raw_state['params']
    return params