"""
Visualizing flattened residual stream

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
import pickle

import matplotlib
from sklearn.decomposition import PCA

import sys
sys.path.append('../')

from model import *

@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 30_000
    res: dict = field(default_factory=dict)
    ds_kwargs: dict = field(default_factory=dict)
    fine_tune_split: float | None = None


# <codecell>
with open('save/cases.pkl', 'rb') as fp:
    all_cases = pickle.load(fp)

case = all_cases[4]
config = case.config

ds, config = CopyDataset.from_config(config)
save_path = case.save_dir
case.name

# <codecell>
mngr = make_ckpt_manager(save_path)
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step())
raw_state = r['state']
params = raw_state['params']

# <codecell>
all_flags = []
for idx in range(2**config.num_layers):
    fmt_str = f'0{config.num_layers}b'
    flags = list(reversed(format(idx, fmt_str)))
    flags = [True if flag == '1' else False for flag in flags]
    all_flags.append(flags)

all_flags = np.array(all_flags)
idxs = np.arange(config.num_layers).reshape(1, -1)
# TODO: pair indices with layers
# %%
