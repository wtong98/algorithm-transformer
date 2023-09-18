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


def evaluate_acc(length, params, config, n_examples=100, use_tqdm=False):
    kwargs = config.ds_generator_kwargs.copy({'lengths': length})
    # TODO: test
    config = config.replace(ds_generator_kwargs=kwargs, vocab_size=params['Embed_0']['embedding'].shape[0])
    train_ds, config = CopyDataset.from_config(config, unify_config=False)

    print('CONFIG', config)
    print('PARAMS', jax.tree_map(lambda x: x.shape, params))

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
            # pred = np.zeros((5,))

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
n_symbols = 100_000_000
test_every = 1
n_test_examples = 32
max_train_len = 5
max_test_len = 15
max_item_label = 50


@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 30_000
    res: dict = field(default_factory=dict)
    ds_kwargs: dict = field(default_factory=dict)
    fine_tune_split: float | None = None

def init_common_kwargs():
    return FrozenDict(
        lengths=tuple(range(1, max_train_len+1)),
        n_nonterminals=max_test_len,
        n_terminals=n_symbols,
        t_lengths=3,
        seed=new_seed()
    )


save_prefix = 'save/'
scratch_dir = os.getenv('SCRATCH')
if scratch_dir is not None:
    save_prefix = scratch_dir +  '/pehlevan_lab/Lab/wlt/transformer/'
    prefix_path = Path(save_prefix)
    if not prefix_path.exists():
        prefix_path.mkdir(parents=True)


all_cases = []
for i in range(n_iters):
    all_cases.extend([
        # Case('NoPE', config=TransformerConfig(
        #     nope_embeding=True), save_dir=f'save/nope_{i}'),
        # Case('Sinusoid', config=TransformerConfig(), save_dir=f'save/sinusoid_{i}'),
        # Case('Relative', config=TransformerConfig(
        #     rel_pos_att=True), save_dir=f'save/relative_{i}'),
        # Case('Random (Relative)', config=TransformerConfig(
        #     rel_pos_att=True, rel_pos_rand_max=(2*max_item_label+2)), save_dir=f'save/relative-rand_{i}'),

        Case('Within (p=0)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=0, **init_common_kwargs())
        ), save_dir=f'cfg_within_0_{i}'),

        Case('Within (p=0.25)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=0.25, **init_common_kwargs())
        ), save_dir=f'cfg_within_0.25_{i}'),

        Case('Within (p=0.5)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=0.5, **init_common_kwargs())
        ), save_dir=f'cfg_within_0.5_{i}'),

        Case('Within (p=0.75)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=0.75, **init_common_kwargs())
        ), save_dir=f'cfg_within_0.75_{i}'),

        Case('Within (p=1)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=1, **init_common_kwargs())
        ), save_dir=f'cfg_within_1_{i}'),

        Case('Cross (p=0)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=0, **init_common_kwargs())
        ), save_dir=f'cfg_cross_0_{i}'),

        Case('Cross (p=0.25)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=0.25, **init_common_kwargs())
        ), save_dir=f'cfg_cross_0.25_{i}'),

        Case('Cross (p=0.5)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=0.5, **init_common_kwargs())
        ), save_dir=f'cfg_cross_0.5_{i}'),

        Case('Cross (p=0.75)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=0.75, **init_common_kwargs())
        ), save_dir=f'cfg_cross_0.75_{i}'),

        Case('Cross (p=1)', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=1, **init_common_kwargs())
        ), save_dir=f'cfg_cross_1_{i}'),

        # Case('base', config=TransformerConfig(
        #     nope_embeding=True,
        #     num_layers=3,
        #     ds_generator_name='RandomGenerator',
        #     ds_generator_kwargs=FrozenDict(lengths=(1,2,3), unique=True, ordered=True, alphabet_size=3)
        # ), save_dir='base')
    ])

for case in all_cases:
    case.save_dir = save_prefix + case.save_dir

# <codecell>
for case in all_cases:
    if Path(case.save_dir).exists():
        print('SKIPPING', case.name)
        continue

    print('TRAINING', case.name)

    init_params = None
    if case.fine_tune_split is not None:
        print('(training base)')
        train_ds, case.config = GenerativeDataset.from_config(case.config)
        train_dl = to_dataloader(train_ds, batch_size=32, pin_memory=True)

        n_iters = int(case.fine_tune_split * case.train_iters)
        state, info = train(case.config, train_dl, eval_dl=train_dl, n_iters=n_iters, print_every=1000)
        init_params = state.params

    train_ds, case.config = CopyDataset.from_config(case.config)
    train_dl = to_dataloader(train_ds, batch_size=32,
                             num_workers=0, pin_memory=True)

    _, info = train(case.config, train_dl, init_params=init_params, eval_dl=train_dl,
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
    case.res['rand_acc'] = []
    for ex_len in tqdm(reversed(range(1, max_test_len + 1, test_every)), total=max_test_len//test_every):
        acc, fails = evaluate_acc(ex_len, params, case.config, n_examples=n_test_examples)

        rand_config = case.config.replace(
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict({
                'alphabet_size': 10   # NOTE: arbitrarily chosen
            })
        )

        rand_acc, _ = evaluate_acc(ex_len, params, rand_config, n_examples=n_test_examples)

        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})
        case.res['rand_acc'].append({'len': ex_len, 'acc': rand_acc})


    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads

# <codecell>
with open('save/cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster

# <codecell>
with open('save/remote/cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

# <codecell>
case = all_cases[0]
case.save_dir = 'save/cfg_within_0_0'
mngr = make_ckpt_manager(case.save_dir)
config = case.config
r = mngr.restore(mngr.best_step())
params = r['state']['params']
print('BEST', mngr.best_step())


evaluate_acc(3, params, config, n_examples=10)

# <codecell>
with open('save/cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

# <codecell>
all_df = []
for case in all_cases:
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
plt.savefig('fig/gen_cfg_overlap_general.png')


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
