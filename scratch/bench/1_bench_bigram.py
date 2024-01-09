"""
Training the model on a simple bigram language model to observe generalization
performance as a function of bigram entropy.

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

n_iters = 5

max_train_len = 10
max_test_len = 25
alphabet_size = max_test_len

train_iters = 30_000

batch_size = 128

betas = [1, 4, 16, 64, 256]

def init_common_kwargs(alphabet_size=alphabet_size):
    return FrozenDict(
        lengths=tuple(range(1, max_train_len+1)),
        alphabet_size=alphabet_size,
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
    random_case = [
        Case('Random', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(**init_common_kwargs())
        ), save_dir=f'random_{i}')
    ]

    bigram_cases = [
        Case(f'Bigram (beta={b})', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='BigramGenerator',
            ds_generator_kwargs=FrozenDict(beta=b, **init_common_kwargs()),
        ), save_dir=f'bigram_{b}_{i}')
    for b in betas]

    structured_case = [
        Case('Uniq and Ord', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(ordered=True, unique=True, **init_common_kwargs(max_test_len))
        ), save_dir=f'ord_and_uniq_{i}')
    ]

    same_case = [
        Case('Same', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(**init_common_kwargs(1))
        ), save_dir=f'count_{i}')
    ]

    all_cases.extend(random_case + bigram_cases + structured_case + same_case)


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

    case.res['acc_in_dist'] = []
    case.res['acc_count'] = []
    case.res['acc_random'] = []

    for ex_len in tqdm(reversed(range(1, max_test_len + 1)), total=max_test_len):

        count_config = case.config.replace(
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(**init_common_kwargs(1))
        )

        random_config = case.config.replace(
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(**init_common_kwargs())
        )

        case.res['acc_in_dist'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, case.config)})

        case.res['acc_count'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, count_config)})

        case.res['acc_random'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, random_config)})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads


# <codecell>
with open('save/bigram_cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster

# <codecell>
# with open('save/cases.pkl', 'rb') as fp:
with open('save/remote/bigram_cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

# <codecell>
def to_df(key):
    all_df = []
    for case in all_cases:
        curr_df = pd.DataFrame(case.res[key])
        curr_df['name'] = case.name
        all_df.append(curr_df)
    df = pd.concat(all_df)
    return df

def plot_bench(df):
    plt.gcf().set_size_inches(28, 3)
    g = sns.boxplot(df, x='len', y='acc', hue='name')
    g.legend_.set_title('')

    g.axvline(9.5, color='red', linestyle='dashed')

    plt.tight_layout()

plot_bench(to_df('acc_in_dist'))
plt.savefig('fig/bigram_acc_in_dist.png')
plt.show()

plot_bench(to_df('acc_count'))
plt.savefig('fig/bigram_acc_count.png')
plt.show()

plot_bench(to_df('acc_random'))
plt.savefig('fig/bigram_acc_random.png')
plt.show()