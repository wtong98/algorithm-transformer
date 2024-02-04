"""
Experimentation with wikitext dataset: https://huggingface.co/datasets/wikitext

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import pickle

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')

from bench_common import *
from model import *

# run_id = new_seed()
run_id = 1130 # NOTE: test seed (should use array parameter)
print('RUN ID', run_id)

n_iters = 1
max_train_len = 10
max_test_len = 25
train_iters = 100_000
batch_size = 128

# n_iters = 1
# max_train_len = 3
# max_test_len = 5
# train_iters = 1_000
# batch_size = 128


save_prefix = 'save/'
cache_dir = None

scratch_dir = os.getenv('SCRATCH')
if scratch_dir is not None:
    save_prefix = scratch_dir +  '/pehlevan_lab/Lab/wlt/transformer/'
    prefix_path = Path(save_prefix)
    if not prefix_path.exists():
        prefix_path.mkdir(parents=True)
    
    cache_dir = save_prefix + 'huggingface_cache'


def init_common_kwargs():
    return FrozenDict(
        lengths=tuple(range(1, max_train_len+1)),
        cache_dir=cache_dir
    )

common_configs = {
    'nope_embedding': True
}


all_cases = []

for i in range(n_iters):
    all_cases.extend([
        Case('Wikitext', config=TransformerConfig(
            ds_generator_name='WikitextGenerator',
            ds_generator_kwargs=FrozenDict(**init_common_kwargs()),
            num_layers=6,
            emb_dim=1024,
            mlp_dim=1024,
            **common_configs,
        ), save_dir=f'wikitext_{run_id}')
    ])


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

    case.res['acc_train'] = []
    case.res['acc_test'] = []
    case.res['acc_random'] = []

    for ex_len in tqdm(reversed(range(1, max_test_len + 1)), total=max_test_len):
        random_config = case.config.replace(
            ds_generator_name='RandomGenerator',
            ds_generator_kwargs=FrozenDict(special_token_override=50256, n_symbols=50257, alphabet_size=50256)  # hardcoded from GPT-2 tokenizer
        )

        test_config = case.config.replace(
            ds_generator_kwargs=FrozenDict({'split': 'test'})
        )

        case.res['acc_train'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, case.config)})

        case.res['acc_test'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, test_config)})

        case.res['acc_random'].append({'len': ex_len, 'acc': 
                                        evaluate_acc(ex_len, params, random_config)})

    jax.clear_caches()  # NOTE: jax currently leaks a lot of threads


# <codecell>
with open(f'save/case_wikitext.{run_id}.pkl', 'wb') as fp:
    pickle.dump(all_cases, fp)

if scratch_dir is not None:
    sys.exit(0)  # terminate now if on cluster


# <codecell>
tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:50000]')
# <codecell>
end_tok = tokenizer.vocab['<|endoftext|>']

def to_tok(ex):
    return tokenizer(ex['text'])

def to_copy_ex(ex, max_len=10):
    probs = 1 / np.arange(1, max_len + 1)
    probs = probs / np.sum(probs)
    length = 1 + np.random.choice(max_len, p=probs)
    sample = ex['input_ids'][:length]
    ex['pattern'] = sample
    return ex

# dataset = dataset.map(to_tok, batched=True)

# <codecell>
dataset = dataset.to_iterable_dataset(num_shards=128) \
                 .map(to_tok, batched=True).map(to_copy_ex) \
                 .filter(lambda row: len(row['pattern']) > 0) \
                 .shuffle(buffer_size=1024)

def to_gen(dataset):
    it = iter(dataset)

    while True:
        try:
            yield next(it)
        except StopIteration:
            print('========== SHUFFLE! ===========')
            dataset = dataset.shuffle()
            it = iter(dataset)


# print(dataset[1]['text'])
# out = tokenizer('dancing under moon-lightttt')
# out

# <codecell>
ds = CopyDataset(to_gen(dataset), special_token_override=end_tok)
it = iter(ds)

# <codecell>

for _ in range(100):
    print(next(it))

# <codecell>
tokenizer.vocab_size

# <codecell>
config = TransformerConfig(
    num_layers=6,
    emb_dim=512,
    mlp_dim=512,
    nope_embedding=True,
    ds_generator_name='WikitextGenerator',
    ds_generator_kwargs=FrozenDict({'lengths': tuple(np.arange(1, 11))})
)

ds, config = CopyDataset.from_config(config)
train_dl = to_dataloader(ds, batch_size=128, pin_memory=True)

state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=20_000, print_every=1_000, save_dir='scratch/save/tmp')

# <codecell>
tokenizer.decode([8498,  2840, 50256,  8498,  2840])

