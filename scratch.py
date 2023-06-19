"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>

from model import *
from task import *

def evaluate_acc(model, length):
    pass # TODO: implement <-- STOPPED HERE


n_symbols = 2
max_item_label = 50

config = TransformerConfig(n_symbols + 3, deterministic=True, use_label_embed=True, max_item_label=max_item_label)
train_ds = CopyDataset(range(1, 10+1), vocab_size=n_symbols, max_item_label=max_item_label)

# config = TransformerConfig(n_symbols + 3, deterministic=True, use_label_embed=False)
# train_ds = CopyDataset(range(1, 10+1), vocab_size=n_symbols)

train_dl = to_dataloader(train_ds, batch_size=32, num_workers=0, pin_memory=True)

# <codecell>
state, info = train(config, train_dl, eval_dl=train_dl, n_iters=3_000, print_every=1_000, save_dir='save/tmp')

# <codecell>
# TODO: make plots
train = stack_forest(info['train_metrics'])
test = stack_forest(info['eval_metrics'])

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for ax, metrics in zip(axs, [train, test]):
    ax.plot(metrics['accuracy'], color='C0', label='accuracy', alpha=0.8)
    ax.set_ylabel('Accuracy', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    # ax.set_xscale('log')

    ax2 = ax.twinx()
    ax2.plot(metrics['loss'], color='C1', label='loss', alpha=0.8)
    ax2.set_ylabel('Loss', color='C1')
    ax2.tick_params(axis='y',labelcolor='C1')

    # ax.plot(metrics['confidence'], label='confidence')
    # ax.plot(metrics['loss'], label='loss')

# plt.savefig('fig/sinus_loss_curve.png')

# <codecell>
mngr = make_ckpt_manager('save/tmp')
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step(), items={'state': None, 'config': TransformerConfig(0)})
raw_state = r['state']

# %%
pred_config = config.replace(deterministic=True)
inputs = [3,3,4,3,3,4,3,3,4,4,3,3,4,4,3,4,1]
predict_with_lab(raw_state['params'], jnp.array(inputs), pred_config, train_ds.tok_to_idx['END'])

# %%
def get_attn_weights(seq, params, config, labels=None):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = TransformerLM(config)
        _, intm = m.apply({'params': params}, seq.reshape(1, -1), labels=labels, mutable='intermediates')
        attn_weights = intm['intermediates']['Decoder'][f'TransformerBlock_{i}']['SingleHeadSelfAttention_0']['attention_weights'][0]
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
        

def plot_sequence(in_seq, state, config):
    seq, labs = predict_with_lab(state, jnp.array(in_seq), pred_config, train_ds.tok_to_idx['END'])
    # seq = jnp.array([3,3,4,3,4,1,3,3,4,3,4])
    print('SEQ', seq)
    attn_weights = get_attn_weights(seq, state, config, labels=labs)
    plot_attn_weights(attn_weights, seq, train_ds.idx_to_tok)

# plot_sequence([3,3,4,3,4,4,4,3,3,4,3,4,3,3,4,3,3,4,1], raw_state['params'], pred_config)
plot_sequence(inputs, raw_state['params'], pred_config)
# plt.savefig('fig/sinus_21.png')

# <codecell>
m = TransformerLM(pred_config)
# _, intm = m.apply({'params': state.params}, jnp.array([3,3,4,3,3,4,1,3,3,4,3,3,4]).reshape(1, -1), mutable='intermediates')
# _, intm = m.apply({'params': state.params}, jnp.array([3,3,4,3,3,1,3,3,4,3,3]).reshape(1, -1), mutable='intermediates')
_, intm = m.apply({'params': state.params}, jnp.array([3,3,3,3,3,3,3,3,3,1,3,3]).reshape(1, -1), mutable='intermediates')
x = intm['intermediates']['Decoder']['TransformerBlock_0']['pre_attention'][0]
x_out = intm['intermediates']['Decoder']['TransformerBlock_0']['post_attention'][0]
mask = intm['intermediates']['Decoder']['TransformerBlock_0']['mask'][0]
pos = intm['intermediates']['Decoder']['PositionEmb']['pos'][0]


att = state.params['Decoder']['TransformerBlock_0']['SelfAttention_0']
wq = att['query']['kernel']
wk = att['key']['kernel']
wv = att['value']['kernel']
w_out = att['out']['kernel']

# jax.tree_map(lambda x: x.shape, state.params)
x.shape
wq.shape

# x = jnp.ones(x.shape)

query = jnp.einsum('...lf,fhd->...lhd', x, wq)
key = jnp.einsum('...lf,fhd->...lhd', x, wk)
value = jnp.einsum('...lf,fhd->...lhd', x, wv)

depth = query.shape[-1]
query /= jnp.sqrt(depth)
attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

attn_weights = jnp.where(mask, attn_weights, -99999)
attn_weights = jax.nn.softmax(attn_weights)
attn_out = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

attn_out = jnp.einsum('...lhd,hdf->...lf', attn_out, w_out)
attn_weights.shape
# jnp.sum(attn_out == x_out)

# QK = wq.squeeze() @ wk.squeeze().T
qk = query.squeeze() @ key.squeeze().T

# qk = x.squeeze() @ x.squeeze().T
qk = jnp.where(mask.squeeze(), qk, -99999)
qk = jax.nn.softmax(qk)
plt.imshow(qk)

# plt.imshow(attn_weights[0,0])
# plt.gca().set_xticks(np.arange(9))
# plt.gca().set_yticks(np.arange(9))
# plt.gca().set_xticklabels([ 'a', 'a', 'b', 'a', 'GO',  'a', 'a', 'b', 'a', ])
# plt.gca().set_yticklabels([ 'a', 'a', 'b', 'a', 'GO',  'a', 'a', 'b', 'a', ])
# plt.savefig('fig/tmp_attention.png')
# attn_weights.shape





# %%
