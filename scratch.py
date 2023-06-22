"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import pickle
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


n_symbols = 2
max_test_len = 50
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
    Case('Item-Label Fixed', config=TransformerConfig(
        vocab_size=n_symbols + 3, max_item_label=max_item_label, freeze_embedding=True, sinus_embedding=True,
    ), save_dir='save/item_label_fixed'),
    Case('Item-Label', config=TransformerConfig(
        vocab_size=n_symbols +3, max_item_label=max_item_label), save_dir='save/item_label'),
    Case('Sinusoid', config=TransformerConfig(
        vocab_size=n_symbols + 3), save_dir='save/sinusoid'),

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
    for ex_len in tqdm(reversed(range(1, max_test_len + 1)), total=max_test_len):
        acc, fails = evaluate_acc(ex_len, params, case.config, max_item_label)
        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})
        case.res['fails'].append({'len': ex_len, 'examples': fails})

# <codecell>
# TODO: remove non-serializable fields (stop gap)
for case in all_cases:
    case.config = case.config.replace(
        kernel_init=None,
        bias_init=None
    )

with open('save/cases_3.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

for case in all_cases:
    case.config = case.config.replace(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6)
    )

# <codecell>
with open('save/cases.pkl', 'rb') as fp:
    c = pickle.load(fp)
c
# <codecell>
all_df = []
for case in all_cases:
    curr_df = pd.DataFrame(case.res['gen_acc'])
    curr_df['name'] = case.name
    all_df.append(curr_df)
df = pd.concat(all_df)

# <codecell>
plt.gcf().set_size_inches(12, 2)
sns.barplot(df, x='len', y='acc', hue='name')
plt.savefig('fig/generalization_acc_3.png')


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

for case in all_cases:
    print('PLOTTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    seq = [3, 3, 4, 4, 3, 4, 3, 3, 4, 3, 3, 4, 4, 3, 4, 1]
    plot_sequence(seq, params, case.config)
    plt.savefig(f'fig/{case.name}_attn_15.png')
