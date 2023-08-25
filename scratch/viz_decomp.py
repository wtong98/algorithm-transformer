"""
Visualizing flattened residual stream

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import sys
sys.path.append('../')

from model import *

@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 30_000
    res: dict = field(default_factory=dict)
    ds_kwargs: dict = field(default_factory=dict)
    fine_tune_split: float | None = None


def plot_path(seq, att_mat, logit_mat, figsize=(18, 6)):
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    im = axs[0].imshow(att_mat.T, vmax=1, vmin=0)
    axs[0].set_xticks(np.arange(len(seq)))
    axs[0].set_xticklabels(seq)
    axs[0].set_xlabel('key')
    axs[0].set_yticks(np.arange(len(seq)))
    axs[0].set_yticklabels(seq)
    axs[0].set_ylabel('query')
    axs[0].set_title('attention')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(logit_mat.T)
    axs[1].set_yticks(np.arange(len(seq)))
    axs[1].set_yticklabels(seq)
    axs[1].set_xlabel('logits')
    axs[1].set_ylabel('sequence')
    axs[1].set_title('value')
    fig.colorbar(im, ax=axs[1])

    prod = logit_mat @ att_mat
    im = axs[2].imshow(prod.T)
    axs[2].set_yticks(np.arange(len(seq)))
    axs[2].set_yticklabels(seq)
    axs[2].set_xlabel('logits')
    axs[2].set_ylabel('sequence')
    axs[2].set_title('logits')
    fig.colorbar(im, ax=axs[2])

    fig.tight_layout()

# <codecell>
subdir = ''
with open(f'save/{subdir}cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

case = all_cases[1]
config = case.config

ds, config = CopyDataset.from_config(config)
name = Path(case.save_dir).name
save_dir = f'save/{subdir}{name}'

case.name

# <codecell>
params = load_params(save_dir)
jax.tree_map(lambda x: x.shape, params)

# <codecell>
inputs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]

seq, info = predict_no_lab(inputs, params, config)
seq

# <codecell>
save_fig_dir = Path('fig/1k_sym')
if not save_fig_dir.exists():
    save_fig_dir.mkdir()

seq = seq

all_flags = []
for idx in range(2**config.num_layers):
    fmt_str = f'0{config.num_layers}b'
    flags = list(format(idx, fmt_str))
    flags = [True if flag == '1' else False for flag in flags]
    all_flags.append(flags)

all_flags = np.array(all_flags)
idxs = np.arange(config.num_layers)

seq = jnp.array(seq)
logits, intm = Transformer(config).apply({'params': params}, seq.reshape(1, -1), mutable='intermediates')
jax.tree_map(lambda x: x.shape, intm)


all_V = []
all_A = []
for i in range(config.num_layers):
    V = params[f'TransformerBlock_{i}']['SingleHeadSelfAttention_0']['value']['kernel']
    A = intm['intermediates'][f'TransformerBlock_{i}']['SingleHeadSelfAttention_0']['attention_weights'][0].squeeze()

    all_V.append(V)
    all_A.append(A)

W_emb = params['Embed_0']['embedding']
W_out = params['LogitDense']['kernel']
b_out = params['LogitDense']['bias']

X_one_hot = jax.nn.one_hot(seq, W_emb.shape[0])
X = (X_one_hot @ W_emb).T

all_logits = []
all_layer_names = []

for flag in tqdm(all_flags):
    idx = idxs[flag]
    pre = [W_out.T] + [all_V[i].T for i in reversed(idx)] + [X]
    post = [all_A[i].T for i in idx]
    layers = ''.join(idx.astype(str))
    all_layer_names.append(layers)

    pre_mat = jnp.linalg.multi_dot(pre)

    post_diff = 2 - len(post)
    if post_diff > 0:
        post = post_diff * [np.eye(X.shape[1])] + post
    post_mat = jnp.linalg.multi_dot(post)

    plt.clf()
    plot_path(seq, post_mat, pre_mat, figsize=(24, 6))
    plt.savefig(save_fig_dir / f"path_{layers}")
    plt.close()

    all_logits.append(pre_mat @ post_mat)

all_logits = jnp.stack(all_logits)
all_logits.shape

accum_logits = jnp.sum(all_logits, axis=0) + b_out.reshape(-1, 1)
assert jnp.all(jnp.isclose(logits.squeeze().T, accum_logits, atol=1e-4))
print('success!')

# %%
max_tok = max(seq)

for i, tok in tqdm(enumerate(seq)):
    plt.clf()

    plt.gcf().set_size_inches(4, 16)
    plt.imshow(all_logits[:,:(max_tok+1),i])
    ax = plt.gca()
    ax.set_xticks(np.arange(max_tok + 1))
    # ax.set_xticklabels(seq)
    ax.set_yticks(np.arange(len(all_layer_names)))
    ax.set_yticklabels(all_layer_names)

    ax.set_title(f'Token: {tok}')
    ax.set_xlabel('Next token')
    ax.set_ylabel('Layers')

    plt.colorbar()
    plt.gcf().tight_layout()
    plt.savefig(save_fig_dir / f"logits_{i}")


# %%
