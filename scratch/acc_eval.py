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
    kwargs = case.config.ds_generator_kwargs.copy({'lengths': length})
    config = case.config.replace(ds_generator_kwargs=kwargs)
    train_ds, config = CopyDataset.from_config(config)

    # train_dl = to_dataloader(train_ds, batch_size=n_examples)

    # batch = next(iter(train_dl))['inputs']
    # prompt = batch[:,:(length+2)]
    # print('PROMPT', prompt)

    # seed = new_seed()
    # rng = jax.random.PRNGKey(seed)

    # m = Transformer(config)
    # for _ in range(length + 1):
    #     rng, curr_rng = jax.random.split(rng)
    #     logits = m.apply({'params': params}, prompt, rngs={'rng': curr_rng})
    #     nxt_toks = logits.argmax(-1)[:,[-1]]
    #     prompt = jnp.concatenate((prompt, nxt_toks), axis=1)
    #     print('PROMPT', prompt)
    
    # aon_acc = jnp.mean(prompt == batch, axis=1)
    # aon_acc = jnp.mean(jnp.isclose(aon_acc, 1))
    # return aon_acc

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


n_iters = 1
n_symbols = 10
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

common_ds_kwargs = FrozenDict(
    lengths=tuple(range(1, max_train_len+1)),
    n_nonterminals=max_test_len,
    # n_terminals=n_symbols
    t_lengths=3
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

        Case('5 Sym', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(n_terminals=5, **common_ds_kwargs)
        ), save_dir=f'cfg_5term_{i}'),

        # Case('10 Sym', config=TransformerConfig(
        #     nope_embeding=True,
        #     ds_generator_name='CfgGenerator',
        #     ds_generator_kwargs=FrozenDict(n_terminals=10, **common_ds_kwargs)
        # ), save_dir=f'cfg_10term_{i}'),

        # Case('50 Sym', config=TransformerConfig(
        #     nope_embeding=True,
        #     ds_generator_name='CfgGenerator',
        #     ds_generator_kwargs=FrozenDict(n_terminals=50, **common_ds_kwargs)
        # ), save_dir=f'cfg_50term_{i}'),

        # Case('100 Sym', config=TransformerConfig(
        #     nope_embeding=True,
        #     ds_generator_name='CfgGenerator',
        #     ds_generator_kwargs=FrozenDict(n_terminals=100, **common_ds_kwargs)
        # ), save_dir=f'cfg_100term_{i}'),

        Case('1000 Sym', config=TransformerConfig(
            nope_embeding=True,
            ds_generator_name='CfgGenerator',
            ds_generator_kwargs=FrozenDict(n_terminals=1000, **common_ds_kwargs)
        ), save_dir=f'cfg_1000term_{i}'),

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
    # if Path(case.save_dir).exists():
    #     print('SKIPPING', case.name)
    #     continue

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
    case.res['fails'] = []
    for ex_len in tqdm(reversed(range(1, max_test_len + 1, test_every)), total=max_test_len//test_every):
        acc, fails = evaluate_acc(ex_len, params, case.config, n_examples=n_test_examples)
        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads
    

# <codecell>
with open('save/cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster


# <codecell>
case = all_cases[1]
mngr = make_ckpt_manager(case.save_dir)
config = case.config
r = mngr.restore(mngr.best_step())
params = r['state']['params']
print('BEST', mngr.best_step())

# evaluate_acc(300, params, config, n_examples=32)
# prompt = [5, 4, 5, 5, 5, 5, 1]
# pred, labs = predict(prompt, params, config)
# correct = np.concatenate((prompt, prompt[1:]))
# # print('corr', correct)
# print('pred', pred)
# print('labs', labs)

evaluate_acc(20, params, config, n_examples=32)


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
# plt.savefig('fig/gen_cfg_symbols_long.png')


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
