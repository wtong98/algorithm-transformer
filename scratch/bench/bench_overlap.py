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

n_iters = 3
n_symbols = 100_000_000
test_every = 1
n_test_examples = 32
max_train_len = 5
max_test_len = 15
max_item_label = 50

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
param_points = [0, 0.25, 0.5, 0.75, 1]
for i in range(n_iters):
    all_cases.extend([

        Case(f'Within (p={p})', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(within_overlap_prob=p, **init_common_kwargs()),
        ), save_dir=f'cfg_within_{p}_{i}')

    for p in param_points])

    all_cases.extend([

        Case(f'Cross (p={p})', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(cross_overlap_prob=p, **init_common_kwargs()),
        ), save_dir=f'cfg_cross_{p}_{i}')

    for p in param_points])

for case in all_cases:
    case.save_dir = save_prefix + case.save_dir

# <codecell>
run_train(all_cases, skip_existing=False)

# <codecell>
for case in all_cases:
    print('TESTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    case.res['same_acc'] = []
    case.res['rand_acc'] = []
    for ex_len in tqdm(reversed(range(1, max_test_len + 1, test_every)), total=max_test_len//test_every):
        acc, fails = evaluate_acc(ex_len, params, case.config, n_examples=n_test_examples)

        rand_config = case.config.replace(
            ds_generator_kwargs=case.config.ds_generator_kwargs.copy({'rand_injection_prob': 1})
        )
        rand_acc, _ = evaluate_acc(ex_len, params, rand_config, n_examples=n_test_examples)

        case.res['same_acc'].append({'len': ex_len, 'acc': acc})
        case.res['rand_acc'].append({'len': ex_len, 'acc': rand_acc})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads

# <codecell>
with open('save/cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster

# <codecell>
# with open('save/cases.pkl', 'rb') as fp:
with open('save/remote/cases.pkl', 'rb') as fp:
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
# plt.savefig('fig/gen_cfg_overlap_general.png')
