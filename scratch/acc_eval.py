"""
Some simple experimentation and visualizations

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import pickle
from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')

from model import *
from task.string_copy_enc import *


def evaluate_acc(length, params, config, max_item_label=-1, n_symbols=2, n_examples=100, use_tqdm=False):
    ds_class = CopyDataset if config.causal else CopyEncDataset
    train_ds = ds_class(length, vocab_size=n_symbols,
                           max_item_label=max_item_label)

    n_correct = 0
    fails = []

    it = zip(range(n_examples), iter(train_ds))
    if use_tqdm:
        it = tqdm(it, total=n_examples)

    if not config.causal:
        dl = to_dataloader(train_ds, batch_size=n_examples)
        batch = next(iter(dl))
        rng = jax.random.PRNGKey(new_seed())

        logits = Transformer(config).apply({'params': params}, batch['inputs'], rngs={'rng': rng})
        preds = jnp.argmax(logits, -1)
        mid = preds.shape[1] // 2
        return jnp.mean(preds[:,mid:] == batch['outputs'][:,mid:]).item(), None

    for _, example in it:

        if config.causal:
            ans = example[0]
            prompt = ans[:len(ans)//2]
        else:
            prompt = example[0].astype('int')
            ans = example[1]

        try:
            # TODO: generalize predict function
            pred = predict_all(prompt, params, config, raw_prompt=True)
        except Exception as e:
            print('failed to predict: ', e)
            # fails.append((prompt, None))
            continue

        if hasattr(pred, '__len__'):
            pred = pred[0]

        # TODO: combine per-token and aon accuracies
        # if pred.shape == ans.shape and np.all(pred == ans):
        #     n_correct += 1
        # else:
        #     pass
        #     fails.append((prompt, pred))
        n_correct += np.mean(pred == ans).item()

    return n_correct / n_examples, fails


n_iters = 3
n_symbols = 2
test_every = 5
max_test_len = 60
max_item_label = 200

def config(**kwargs):
    return TransformerConfig(
        vocab_size=n_symbols + 3,
        causal=False,
        **kwargs
    )


@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 20_000
    res: dict = field(default_factory=dict)
    train_len_min: int = 1
    train_len_max: int = 5

all_cases = []
for i in range(n_iters):
    all_cases.extend([
        Case('NoPE', config=TransformerConfig(
            vocab_size=n_symbols +3, nope_embeding=True), save_dir=f'save/nope_{i}'),
        Case('Sinusoid', config=TransformerConfig(
            vocab_size=n_symbols + 3), save_dir=f'save/sinusoid_{i}'),
        # Case('Sinusoid (Item-Label)', config=TransformerConfig(
        #     vocab_size=n_symbols + 3, max_item_label=max_item_label, freeze_embedding=True, sinus_embedding=True,
        # ), save_dir=f'save/item-label-fixed_{i}'),
        Case('Relative', config=TransformerConfig(
            vocab_size=n_symbols + 3, rel_pos_att=True), save_dir=f'save/relative_{i}'),
        Case('Random (Relative)', config=TransformerConfig(
            vocab_size=n_symbols +3, rel_pos_att=True, rel_pos_rand_max=(2*max_item_label+2)), save_dir=f'save/relative-rand_{i}'),
        # Case('Random (Item-Label)', config=TransformerConfig(
        #     vocab_size=n_symbols +3, max_item_label=max_item_label), save_dir=f'save/item-label_{i}'),
    ])

for case in all_cases:
    case.config = case.config.replace(causal=False)

# <codecell>
for case in all_cases:
    print('TRAINING', case.name)

    train_ds = CopyEncDataset(range(1, case.train_len_max+1),
                           vocab_size=n_symbols, max_item_label=max_item_label)
    train_dl = to_dataloader(train_ds, batch_size=32,
                             num_workers=0, pin_memory=True)

    _, info = train(case.config, train_dl, eval_dl=train_dl,
                    n_iters=case.train_iters, print_every=1_000, save_dir=case.save_dir)
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
        acc, fails = evaluate_acc(ex_len, params, case.config, max_item_label=max_item_label, n_examples=32)
        case.res['gen_acc'].append({'len': ex_len, 'acc': acc})
        # case.res['fails'].append({'len': ex_len, 'examples': fails})

# <codecell>
mngr = make_ckpt_manager(all_cases[5].save_dir)
config = all_cases[0].config
r = mngr.restore(mngr.best_step())
params = r['state']['params']

# evaluate_acc(300, params, config, n_examples=32)
prompt = list(np.random.choice([3, 4], size=100))
pred, _ = predict_all(prompt, params, config)
correct = np.concatenate((prompt, prompt))
print('correct', correct)
print('pred', pred)



# <codecell>
# TODO: remove non-serializable fields (stop gap)
# for case in all_cases:
#     case.config = case.config.replace(
#         kernel_init=None,
#         bias_init=None
#     )

with open('save/cases.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

# for case in all_cases:
#     case.config = case.config.replace(
#         kernel_init=nn.initializers.xavier_uniform(),
#         bias_init=nn.initializers.normal(stddev=1e-6)
#     )

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
g = sns.lineplot(df, x='len', y='acc', hue='name')
g.legend_.set_title(None)
sns.move_legend(g, 'lower right')

plt.axvline(4.5, color='red', linestyle='dashed')
plt.ylabel('acc (aon)')
plt.gcf().tight_layout()
# plt.savefig('fig/gen_no_causal_aon_long.png')


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
