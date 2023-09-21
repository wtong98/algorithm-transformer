"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import os
from pathlib import Path
import pickle

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../../')

from bench_common import *
from model import *
from task.string_copy import *

n_iters = 10
n_symbols = 100_000_000
test_every = 1
n_test_examples = 32
max_train_len = 5
max_test_len = 15
max_item_label = 50
train_iters = 100_000

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
ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in range(n_iters):
    all_cases.extend([
        Case(f'Rand (p={p})', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(rand_injection_prob=p, **init_common_kwargs()),
        ), save_dir=f'cfg_rand_{p}_{i}')
    for p in ps])

for case in all_cases:
    case.save_dir = save_prefix + case.save_dir
    case.train_iters = train_iters

# <codecell>
run_train(all_cases, skip_existing=True)

# <codecell>
for case in all_cases:
    print('TESTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    case.res['struct_acc'] = []
    case.res['same_acc'] = []
    case.res['rand_acc'] = []
    for ex_len in tqdm(reversed(range(1, max_test_len + 1, test_every)), total=max_test_len//test_every):
        acc = evaluate_acc(ex_len, params, case.config, n_examples=n_test_examples)

        struct_config = case.config.replace(
            ds_generator_kwargs=case.config.ds_generator_kwargs.copy({'rand_injection_prob': 0})
        )
        struct_acc = evaluate_acc(ex_len, params, struct_config, n_examples=n_test_examples)

        rand_config = case.config.replace(
            ds_generator_kwargs=case.config.ds_generator_kwargs.copy({'rand_injection_prob': 1})
        )
        rand_acc = evaluate_acc(ex_len, params, rand_config, n_examples=n_test_examples)

        case.res['struct_acc'].append({'len': ex_len, 'acc': struct_acc})
        case.res['same_acc'].append({'len': ex_len, 'acc': acc})
        case.res['rand_acc'].append({'len': ex_len, 'acc': rand_acc})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads

# <codecell>
with open('save/rand_inj_cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster

# <codecell>
# with open('save/cases.pkl', 'rb') as fp:
with open('save/remote/rand_inj_cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

# <codecell>
# TODO: plot pretty, retrain until convergence, confirm remarkable results
def to_df(key):
    all_df = []
    for case in all_cases:
        curr_df = pd.DataFrame(case.res[key])
        curr_df['name'] = case.name
        all_df.append(curr_df)
    df = pd.concat(all_df)
    return df

keys = ['same_acc', 'struct_acc', 'rand_acc']
all_dfs = [to_df(k) for k in keys]
titles = ['Same', 'Structured', 'Random']

# <codecell>
fig, axs = plt.subplots(3, 1, figsize=(28, 9))
for ax, df, title in zip(axs, all_dfs, titles):
    g = sns.barplot(df, x='len', y='acc', hue='name', ax=ax)
    g.legend_.set_title(None)
    # sns.move_legend(g, 'lower left')

    ax.axvline(4.5, color='red', linestyle='dashed')
    ax.set_ylabel('acc (aon)')
    ax.set_title(title)

plt.gcf().tight_layout()
plt.savefig('fig/gen_cfg_rand_injection_convergence_interleaved.png')

# <codecell>
case = all_cases[8]
case.res

case.save_dir = 'save/cfg_rand_0.8_0'

mngr = make_ckpt_manager(case.save_dir)
r = mngr.restore(mngr.best_step())
params = r['state']['params']

ds, _ = CopyDataset.from_config(case.config)
ds.gen.nt_to_ts
it = iter(ds)


# <codecell>
next(ds)
