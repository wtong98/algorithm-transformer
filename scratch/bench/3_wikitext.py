"""
Experimentation with wikitext dataset: https://huggingface.co/datasets/wikitext

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import sys
sys.path.append('../../')

from bench_common import *
from model import *

tokenizer = AutoTokenizer.from_pretrained('gpt2')
dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train[:50000]')

# <codecell>
end_tok = tokenizer.vocab['<|endoftext|>']

def to_tok(ex):
    return tokenizer(ex['text'])

# TODO: consider subsetting rather than prefixing
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
    vocab_size=tokenizer.vocab_size,
    num_layers=3,
    emb_dim=128,
    mlp_dim=128,
    nope_embedding=True
)


train_dl = to_dataloader(ds, batch_size=128, pin_memory=True)

state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=20_000, print_every=1_000, save_dir='scratch/save/tmp')

# <codecell>
tokenizer.decode([8498,  2840, 50256,  8498,  2840])

