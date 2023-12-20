"""
Training the model on datasets of fixed size and witnessing generalization
performance

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

n_iters=3
n_symbols = np.iinfo(np.int32).max
max_train_len = 10
max_test_len = 30
train_iters = 30_000
batch_size = 128

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
data_sizes = [64, 128, 256, 512, 1024, 2048, None]

for i in range(n_iters):
    all_cases.extend([
        Case(f'CFG (d={d})', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(fix_size=d, **init_common_kwargs()),
        ), save_dir=f'cfg_fixed_{d}_{i}')
    for d in data_sizes])

for case in all_cases:
    case.save_dir = save_prefix + case.save_dir
    case.train_iters = train_iters

# <codecell>
run_train(all_cases, skip_existing=False)


# <codecell>
for case in all_cases:
    print('TESTING', case.name)
    mngr = make_ckpt_manager(case.save_dir)
    r = mngr.restore(mngr.best_step())
    params = r['state']['params']

    case.res['acc'] = []

    for ex_len in tqdm(reversed(range(1, max_test_len + 1)), total=max_test_len):
        acc = evaluate_acc(ex_len, params, case.config)
        case.res['acc'].append({'len': ex_len, 'acc': acc})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads


# <codecell>
with open('save/cfg_fix_cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster

# <codecell>
# with open('save/cases.pkl', 'rb') as fp:
with open('save/remote/cfg_fix_cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)


# <codecell>

config = TransformerConfig(
    num_layers=3,
    nope_embeding=True,

    ds_generator_name='CfgGenerator',
    ds_generator_kwargs=FrozenDict({
        'lengths': tuple(np.arange(max_train_len) + 1),
        't_lengths': 3,
        'sampling_strategy': 'zipf',
        'n_nonterminals': max_len,
        'n_terminals': n_symbols,
        'fix_size': 512,
        'seed': 1,
    }))

    # ds_generator_name='RandomGenerator',
    # ds_generator_kwargs=FrozenDict({
    #     'lengths': tuple(np.arange(max_train_len) + 1),
    #     'unique': True,
    #     'ordered': True,
    #     'alphabet_size': n_symbols
    # }))

train_ds, config = CopyDataset.from_config(config)
train_dl = to_dataloader(train_ds, batch_size=128, pin_memory=True)


# <codecell>
state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=5_000, print_every=1_000, save_dir='scratch/save/tmp')

plot_train_metrics(info)



# %%
evaluate_acc(15, state.params, config, use_tqdm=True)