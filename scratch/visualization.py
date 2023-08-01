"""
Visualizing transformer models

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib
from sklearn.decomposition import PCA

import sys
sys.path.append('../')

from model import *

config = TransformerConfig(50 + 4, nope_embeding=True)

mngr = make_ckpt_manager('save/copy_subsets/ord_and_uniq_0')
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step())
raw_state = r['state']
params = raw_state['params']

# <codecell>
pred, _ = predict([3, 4, 5, 6, 7, 8, 9, 10, 1], params, config)
pred

# <codecell>
emb = params['Embed_0']['embedding']
# emb = np.random.randn(54, 128)
pca = PCA().fit(emb)
emb_pca = pca.transform(emb)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

# <codecell>
emb_pcs = emb_pca[:,[0 ,2]]
n_comp = emb_pcs.shape[0]

cmap = matplotlib.colormaps['viridis']
cs = [cmap(idx) for idx in np.arange(n_comp) / n_comp]
sc = plt.scatter(emb_pcs[:,0], emb_pcs[:,1], c=cs)
plt.colorbar(sc)


# <codecell>
voc_emb = emb[4:]
voc_dot = np.einsum('ik,jk->ij', voc_emb, voc_emb)

magn = np.sqrt(np.diag(voc_dot).reshape(-1, 1))
norm = magn @ magn.T

# plt.imshow(norm)

# idxs = np.sort(np.random.choice(50, size=15, replace=False))
plt.plot(np.diag(norm))
cos_score = voc_dot / norm

# plt.imshow(cos_score)
# plt.colorbar()

# <codecell>
q0 = params['TransformerBlock_0']['SingleHeadSelfAttention_0']['query']['kernel']
k0 = params['TransformerBlock_0']['SingleHeadSelfAttention_0']['key']['kernel']

query_emb = voc_emb @ q0
key_emb = voc_emb @ k0

att_dot = np.einsum('ik,jk->ij', query_emb, key_emb)
plt.imshow(att_dot)
plt.colorbar()


# <codecell>
jax.tree_map(lambda x: x.shape, params)





# <codecell>
def get_attn_weights(seq, params, config, labels=None):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = Transformer(config)
        _, intm = m.apply({'params': params}, seq.reshape(
            1, -1), labels=labels, mutable='intermediates')
        attn_weights = intm['intermediates'][f'TransformerBlock_{i}'][
            'SingleHeadSelfAttention_0']['attention_weights'][0]
        all_weights.append(attn_weights.squeeze())

    all_weights = jnp.stack(all_weights)
    return all_weights


def plot_attn_weights(attn_weights, seq, idx_to_tok):
    n_layers = attn_weights.shape[0]
    fig, axs = plt.subplots(1, n_layers, figsize=(7 * n_layers, 7))

    if n_layers == 1:
        axs = [axs]

    for i, (attn, ax) in enumerate(zip(attn_weights, axs)):
        ax.imshow(attn)
        ax.set_xticks(np.arange(len(seq)))
        ax.set_xticklabels([idx_to_tok[idx] for idx in seq])
        ax.set_yticks(np.arange(len(seq)))
        ax.set_yticklabels([idx_to_tok[idx] for idx in seq])

        ax.set_xlabel('Token')
        ax.set_ylabel('Time')
        ax.set_title(f'Layer {i+1}')

    fig.tight_layout()


def plot_sequence(in_seq, params, config):
    seq, labs = predict(jnp.array(in_seq), params, config)
    labs = None
    # seq = jnp.array([3,3,4,3,4,1,3,3,4,3,4])
    seq = seq[:(len(in_seq)*2)]
    print('SEQ', seq)
    attn_weights = get_attn_weights(seq, params, config, labels=labs)
    plot_attn_weights(attn_weights, seq, train_ds.idx_to_tok)


train_ds = CopyDataset(10, vocab_size=50)
plot_sequence([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1], params, config)
plt.savefig('fig/tmp_att.png')