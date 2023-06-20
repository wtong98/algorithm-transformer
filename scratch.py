"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>

from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

from model import *
from task import *


def evaluate_acc(length, params, config, max_item_label=-1, n_symbols=2, n_examples=100, use_tqdm=False):
    train_ds = CopyDataset(length, vocab_size=n_symbols,
                           max_item_label=max_item_label)

    n_correct = 0
    fails = []

    it = zip(range(n_examples), iter(train_ds))
    if use_tqdm:
        it = tqdm(it, total=n_examples)

    for _, example in it:
        ans = example[0]
        prompt = ans[:len(ans)//2]
        pred, _ = predict(params, prompt, config, train_ds.tok_to_idx['END'])

        if pred.shape == ans.shape and np.all(pred == ans):
            n_correct += 1
        else:
            fails.append((prompt, pred))

    return n_correct / n_examples, fails


# <codecell>
n_symbols = 2
max_item_label = 50

config = TransformerConfig(
    n_symbols + 3, deterministic=True, max_item_label=max_item_label)
train_ds = CopyDataset(range(1, 10+1), vocab_size=n_symbols,
                       max_item_label=max_item_label)
train_dl = to_dataloader(train_ds, batch_size=32,
                         num_workers=0, pin_memory=True)

state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=3_000, print_every=1_000, save_dir='save/tmp')

# <codecell>
evaluate_acc(15, state.params, config,
             max_item_label=max_item_label, n_examples=5)
# <codecell>
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
    ax2.tick_params(axis='y', labelcolor='C1')

    # ax.plot(metrics['confidence'], label='confidence')
    # ax.plot(metrics['loss'], label='loss')

# plt.savefig('fig/sinus_loss_curve.png')

# <codecell>
mngr = make_ckpt_manager('save/tmp')
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step(), items={
                 'state': None, 'config': TransformerConfig(0)})
raw_state = r['state']

# %%
pred_config = config.replace(deterministic=True)
inputs = [3, 3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 4, 1]
predict_with_lab(raw_state['params'], jnp.array(
    inputs), pred_config, train_ds.tok_to_idx['END'])

# %%


def get_attn_weights(seq, params, config, labels=None):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = TransformerLM(config)
        _, intm = m.apply({'params': params}, seq.reshape(
            1, -1), labels=labels, mutable='intermediates')
        attn_weights = intm['intermediates']['Decoder'][f'TransformerBlock_{i}'][
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
    seq, labs = predict(params, jnp.array(
        in_seq), config, train_ds.tok_to_idx['END'])
    # seq = jnp.array([3,3,4,3,4,1,3,3,4,3,4])
    print('SEQ', seq)
    attn_weights = get_attn_weights(seq, params, config, labels=labs)
    plot_attn_weights(attn_weights, seq, train_ds.idx_to_tok)


# plot_sequence([3,3,4,3,4,4,4,3,3,4,3,4,3,3,4,3,3,4,1], raw_state['params'], pred_config)
# plot_sequence(inputs, raw_state['params'], pred_config)
# plt.savefig('fig/sinus_21.png')


# <codecell>
n_symbols = 2
max_test_len = 20
max_item_label = 50


@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 10_000
    res: dict = field(default_factory=dict)
    train_len_min: int = 1
    train_len_max: int = 10


all_cases = [
    Case('Sinusoid', config=TransformerConfig(
        vocab_size=n_symbols + 3), save_dir='save/sinusoid'),
    Case('Item-Label', config=TransformerConfig(vocab_size=n_symbols +
         3, max_item_label=max_item_label), save_dir='save/item_label'),

]

# <codecell>
for case in all_cases:
    print('TRAINING', case.name)

    train_ds = CopyDataset(range(1, case.train_len_max+1),
                           vocab_size=n_symbols, max_item_label=max_item_label)
    train_dl = to_dataloader(train_ds, batch_size=32,
                             num_workers=0, pin_memory=True)

    _, info = train(case.config, train_dl, eval_dl=train_dl,
                    n_iters=case.train_iters, print_every=1_000, save_dir=case.save_dir)
    case.res['train_metrics'] = info['train_metrics']
    case.res['eval_metrics'] = info['eval_metrics']

# <codecell>
for case in all_cases:
    print('TESTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    case.res['gen_acc'] = []
    case.res['fails'] = []
    for ex_len in tqdm(range(1, max_test_len + 1)):
        acc, fails = evaluate_acc(ex_len, params, case.config, max_item_label)
        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})
        case.res['fails'].append({'len': ex_len, 'examples': fails})

# <codecell>
all_df = []
for case in all_cases:
    curr_df = pd.DataFrame(case.res['gen_acc'])
    curr_df['name'] = case.name
    all_df.append(curr_df)
df = pd.concat(all_df)

# <codecell>
plt.gcf().set_size_inches(8, 3)
sns.barplot(df, x='len', y='acc', hue='name')
plt.savefig('fig/generalization_acc.png')


# %%
for case in all_cases:
    print('PLOTTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    seq = [3, 3, 4, 4, 3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 4, 1]
    plot_sequence(seq, params, case.config)
    plt.savefig(f'fig/{case.name}_attn_15.png')
