"""
Visualizing transformer models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
import pickle

import matplotlib
from sklearn.decomposition import PCA

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

# <codecell>
with open('save/cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

case = all_cases[4]
config = case.config

ds, config = CopyDataset.from_config(config)
save_path = case.save_dir
case.name

# <codecell>

mngr = make_ckpt_manager(save_path)
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step())
raw_state = r['state']
params = raw_state['params']

# <codecell>
pred, _ = predict_no_lab([3, 11, 13, 12, 13, 12, 8, 1], params, config)
pred

# <codecell>
emb = params['Embed_0']['embedding']
# emb = np.random.randn(54, 128)
pca = PCA().fit(emb)
emb_pca = pca.transform(emb)

emb_rand = np.random.randn(*emb.shape)
pca_rand = PCA().fit(emb_rand)

plt.plot(np.cumsum(pca.explained_variance_ratio_), label='embeddings')
plt.plot(np.cumsum(pca_rand.explained_variance_ratio_), color='black', alpha=0.7, linestyle='dashed', label='random')

plt.xlabel('PC')
plt.ylabel('Proportion of variance')
# plt.savefig('fig/pca_uniq_ord.png')

# <codecell>
emb_pcs = emb_pca[:,[0 ,2]]
n_comp = emb_pcs.shape[0]

cmap = matplotlib.colormaps['viridis']
cs = [cmap(idx) for idx in np.arange(n_comp) / n_comp]
sc = plt.scatter(emb_pcs[:,0], emb_pcs[:,1], c=cs)
plt.colorbar(sc)


# <codecell>
voc_emb = emb[:]
voc_dot = np.einsum('ik,jk->ij', voc_emb, voc_emb)

magn = np.sqrt(np.diag(voc_dot).reshape(-1, 1))
norm = magn @ magn.T

# plt.imshow(norm)

# idxs = np.sort(np.random.choice(50, size=15, replace=False))
# plt.plot(np.diag(norm))
cos_score = voc_dot / norm

plt.imshow(cos_score)
plt.colorbar()

# <codecell>
q0 = params['TransformerBlock_0']['SingleHeadSelfAttention_0']['query']['kernel']
k0 = params['TransformerBlock_0']['SingleHeadSelfAttention_0']['key']['kernel']
# q0 = np.eye(q0.shape[0])
# k0 = np.eye(k0.shape[0])

query_emb = voc_emb @ q0
key_emb = voc_emb @ k0

att_dot = np.einsum('ik,jk->ij', query_emb, key_emb)
plt.imshow(att_dot[:,:])
plt.colorbar()
# plt.savefig('fig/dot_layer1_aou_1l.png')

# <codecell>
# plt.plot(att_dot[1,4:])
# plt.plot(att_dot[3,4:])
cmap = matplotlib.colormaps['viridis']
cs = [cmap(idx) for idx in np.arange(n_comp) / n_comp]

plt.scatter(att_dot[1,4:], att_dot[3,4:], c=cs[4:])

# <codecell>
pred, _ = predict([3, 4, 5, 6, 7, 8, 9, 10, 1], params, config)
pred


# <codecell>
def get_attn_weights(seq, params, config, labels=None, intm_name='attention_weights'):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = Transformer(config)
        _, intm = m.apply({'params': params}, seq.reshape(
            1, -1), labels=labels, mutable='intermediates')
        attn_weights = intm['intermediates'][f'TransformerBlock_{i}'][
            'SingleHeadSelfAttention_0'][intm_name][0]
        all_weights.append(attn_weights.squeeze())

    all_weights = jnp.stack(all_weights)
    return all_weights


def plot_attn_weights(attn_weights, seq, idx_to_tok, axs=None):
    n_layers = attn_weights.shape[0]
    if axs is None:
        _, axs = plt.subplots(1, n_layers, figsize=(7 * n_layers, 7))

    if n_layers == 1:
        axs = [axs]

    for i, (attn, ax) in enumerate(zip(attn_weights, axs)):
        im = ax.imshow(attn)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(seq)))
        ax.set_xticklabels([idx_to_tok[idx] if idx in (1,2,3) else idx for idx in seq])
        ax.set_yticks(np.arange(len(seq)))
        ax.set_yticklabels([idx_to_tok[idx] if idx in (1,2,3) else idx for idx in seq])

        ax.set_xlabel('Token')
        ax.set_ylabel('Time')
        ax.set_title(f'Layer {i+1}')


def plot_sequence(in_seq, params, config):
    seq, labs = predict(jnp.array(in_seq), params, config)
    labs = None
    # seq = jnp.array([3,3,4,3,4,1,3,3,4,3,4])
    seq = seq[:(len(in_seq)*2)]
    print('SEQ', seq)

    fig, axs = plt.subplots(3, config.num_layers, figsize=(7 * config.num_layers, 21))

    emb = get_attn_weights(seq, params, config, intm_name='inputs')
    emb_dot = np.einsum('bik,bjk->bij', emb, emb)
    plot_attn_weights(emb_dot, seq, train_ds.idx_to_tok, axs=axs[0])

    raw_att = get_attn_weights(seq, params, config, intm_name='raw_att')
    plot_attn_weights(raw_att, seq, train_ds.idx_to_tok, axs=axs[1])

    attn_weights = get_attn_weights(seq, params, config, labels=labs)
    plot_attn_weights(attn_weights, seq, train_ds.idx_to_tok, axs=axs[2])

    fig.tight_layout()


train_ds = ds
plot_sequence([3, 759, 954, 38, 1], params, config)
# plot_sequence([3, 7, 8, 4, 4, 5, 5, 5, 4, 1], params, config)
plt.savefig('fig/att_cfg_1000_sym_tuple.png')
# %%
