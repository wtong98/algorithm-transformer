"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import os
from pathlib import Path
import pickle
from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')

from model import *
from task.string_copy import *


def evaluate_acc(params, config, n_examples=100, use_tqdm=False):
    config, train_ds = CopyDataset.from_config(config)

    n_correct = 0
    fails = []

    it = zip(range(n_examples), iter(train_ds))
    if use_tqdm:
        it = tqdm(it, total=n_examples)

    for _, example in it:
        ans = example[0]
        offset = 1 if train_ds.bos else 0
        prompt = ans[:len(ans)//2+offset]

        try:
            pred = predict(prompt, params, config)

        except Exception as e:
            print('failed to predict: ', e)
            # fails.append((prompt, None))
            continue

        if hasattr(pred, '__len__'):
            pred = pred[0]

        # TODO: combine per-token and aon accuracies
        if pred.shape == ans.shape and np.all(pred == ans):
            n_correct += 1
        else:
            pass
        #     fails.append((prompt, pred))
        # n_correct += np.mean(pred == ans).item()

    return n_correct / n_examples, fails


n_iters = 3
n_symbols = 10
test_every = 1
n_test_examples = 32
max_train_len = 5
max_test_len = 25
max_item_label = 50



@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 30_000
    res: dict = field(default_factory=dict)
    ds_kwargs: dict = field(default_factory=dict)

common_ds_kwargs = FrozenDict(
    nt_lengths=tuple(range(1, max_train_len+1)),
    t_lengths=tuple(range(1, 4)),
    n_nonterminals=max_test_len,
    n_terminals=n_symbols
)


save_prefix = ''
scratch_dir = os.getenv('SCRATCH')
if scratch_dir is not None:
    save_prefix = scratch_dir +  '/pehlevan_lab/Lab/wlt/transformer/'
    prefix_path = Path(save_prefix)
    if not prefix_path.exists():
        prefix_path.mkdir(parents=True)


all_cases = []
for i in range(n_iters):
    all_cases.extend([
        # Case('1 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=1), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_1l_{i}'),
        # Case('2 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=2), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_2l_{i}'),
        # Case('3 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=3), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_3l_{i}'),
        # Case('4 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=4), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_4l_{i}'),
        # Case('5 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=5), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_5l_{i}'),
        # Case('6 Layer (ordered and unique)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True, num_layers=6), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/oau_6l_{i}'),
        
        # Case('Item-Label', config=TransformerConfig(
        #     vocab_size=n_symbols +4, max_item_label=max_item_label), ds_kwargs={}, save_dir=f'save/item_label_{i}'),
        # Case('Ordered and Unique', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/ord_and_uniq_{i}'),
        # Case('Ordered', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True), ds_kwargs={'ordered': True}, save_dir=f'save/ord_{i}'),
        # Case('Unique', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True), ds_kwargs={'unique': True}, save_dir=f'save/uniq_{i}'),
        # Case('Neither', config=TransformerConfig(
        #     vocab_size=n_symbols +4, nope_embeding=True), ds_kwargs={}, save_dir=f'save/neither_{i}'),

        Case('NoPE', config=TransformerConfig(
            nope_embeding=True), save_dir=f'save/nope_{i}'),
        Case('Sinusoid', config=TransformerConfig(), save_dir=f'save/sinusoid_{i}'),
        Case('Relative', config=TransformerConfig(
            rel_pos_att=True), save_dir=f'save/relative_{i}'),
        Case('Random (Relative)', config=TransformerConfig(
            rel_pos_att=True, rel_pos_rand_max=(2*max_item_label+2)), save_dir=f'save/relative-rand_{i}'),
        

        # Case('NoPE', config=TransformerConfig(
        #     vocab_size=n_symbols + 4, nope_embeding=True), save_dir=f'save/nope_{i}', ds_kwargs={}),
        # Case('Sinusoid', config=TransformerConfig(
        #     vocab_size=n_symbols + 4), save_dir=f'save/sinusoid_{i}', ds_kwargs={}),
        # # Case('Sinusoid (Item-Label)', config=TransformerConfig(
        # #     vocab_size=n_symbols + 3, max_item_label=max_item_label, freeze_embedding=True, sinus_embedding=True,
        # # ), save_dir=f'save/item-label-fixed_{i}'),
        # Case('Relative', config=TransformerConfig(
        #     vocab_size=n_symbols + 4, rel_pos_att=True), ds_kwargs={}, save_dir=f'save/relative_{i}'),
        # # Case('Random (Relative)', config=TransformerConfig(
        # #     vocab_size=n_symbols +4, rel_pos_att=True, rel_pos_rand_max=(2*max_item_label+2)), ds_kwargs={'unique': True, 'ordered': True}, save_dir=f'save/relative-rand_{i}'),
        # Case('Random (Item-Label)', config=TransformerConfig(
        #     vocab_size=n_symbols +4, max_item_label=max_item_label), ds_kwargs={'bos': False}, save_dir=f'save/item-label_{i}'),
    ])

for case in all_cases:
    case.save_dir = save_prefix + case.save_dir
    case.config = case.config.replace(
        ds_generator_name='CfgGenerator',
        ds_generator_kwargs=common_ds_kwargs)

# <codecell>
for case in all_cases:
    if Path(case.save_dir).exists():
        print('SKIPPING', case.name)
        continue

    print('TRAINING', case.name)

    train_ds, case.config = CopyDataset.from_config(case.config)
    train_dl = to_dataloader(train_ds, batch_size=32,
                             num_workers=0, pin_memory=True)

    _, info = train(case.config, train_dl, eval_dl=train_dl,
                    n_iters=case.train_iters, print_every=1_000, save_dir=case.save_dir)
    plot_train_metrics(info, save_path=case.save_dir + '/metrics.png')
    # case.res['train_metrics'] = info['train_metrics']
    # case.res['eval_metrics'] = info['eval_metrics']

# <codecell>
for case in all_cases:
    print('TESTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    case.res['gen_acc'] = []
    case.res['fails'] = []
    for ex_len in tqdm(reversed(range(1, max_test_len + 1, test_every)), total=max_test_len//test_every):
        acc, fails = evaluate_acc(ex_len, params, case.config, max_item_label=max_item_label, n_examples=n_test_examples, n_symbols=case.config.vocab_size-4, ds_kwargs=case.ds_kwargs)
        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})
        # case.res['fails'].append({'len': ex_len, 'examples': fails})

# <codecell>
with open('save/cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster


# <codecell>
mngr = make_ckpt_manager(all_cases[3].save_dir)
config = all_cases[3].config
r = mngr.restore(mngr.best_step())
params = r['state']['params']
print('BEST', mngr.best_step())

# evaluate_acc(300, params, config, n_examples=32)
prompt = [5, 4, 5, 5, 5, 5, 1]
pred, labs = predict(prompt, params, config)
correct = np.concatenate((prompt, prompt[1:]))
# print('corr', correct)
print('pred', pred)
print('labs', labs)

evaluate_acc(8, params, config, n_symbols=n_symbols, n_examples=5, ds_kwargs=all_cases[3].ds_kwargs)


# <codecell>
with open('save/remote/cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

# <codecell>
all_df = []
for case in all_cases:
    if not ('neither' in case.name):
        continue 

    curr_df = pd.DataFrame(case.res['gen_acc'])
    curr_df['name'] = case.name
    all_df.append(curr_df)
df = pd.concat(all_df)

# <codecell>
plt.gcf().set_size_inches(28, 3)
g = sns.barplot(df, x='len', y='acc', hue='name')
g.legend_.set_title(None)
sns.move_legend(g, 'lower left')

plt.axvline(4.5, color='red', linestyle='dashed')
plt.ylabel('acc (aon)')
plt.gcf().tight_layout()
plt.savefig('fig/gen_neither.png')


# %%
def get_attn_weights(seq, params, config, labels=None):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = Transformer(config)
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
