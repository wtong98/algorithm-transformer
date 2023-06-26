"""
Experiment with PDFA extraction

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('../')

from model import *
from task import *
from wlstar.Learner import learn

class ParseError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def make_labels(prompt, config):
    prompt_idx, = np.where(prompt == 1) # tok_to_idx['GO'] == 1
    labels = np.sort(
        np.random.choice(
            np.arange(1, config.max_item_label + 1), 
            size=prompt_idx, 
            replace=False)
    )

    n_rest = len(prompt) - prompt_idx.item() - 1
    labels = np.concatenate((
        labels, 
        [0], 
        labels[:n_rest],
        [0] if prompt[-1] == 2 else [])) # tok_to_idx['END] == 2
    return labels
    

def get_probs(prompt, labels, params, config):
    assert len(prompt.shape) == 1
    prompt = prompt.reshape(1, -1)
    labels = labels.reshape(1, -1)

    m = TransformerLM(config)
    logits = m.apply({'params': params}, prompt, labels=labels)
    logits_tok, logits_labs = logits[...,:config.vocab_size], logits[...,config.vocab_size:]
    probs_tok = jax.nn.softmax(logits_tok[[0],-1, 2:])  # normalized over returnable values
    probs_lab = jax.nn.softmax(logits_labs[[0],-1,1:])  # normalized over nonzero labels

    end_tok_prob = probs_tok[0,0]
    sym_tok_probs = probs_tok[:,1:]
    probs = sym_tok_probs.T @ probs_lab
    probs = jnp.append(probs.flatten(), end_tok_prob)

    return probs


class DummyAuto:
    def __init__(self) -> None:
        self.input_alphabet = ['a', 'b']
        self.end_token = 'f'
        self.internal_alphabet = self.input_alphabet + [self.end_token]

    def initial_state(self):
        return ('s',)

    def next_state(self, curr_state, token):
        return curr_state + (token,)

    def state_probs_dist(self, state):
        if state[-1] == 's':
            return [0.5, 0.5, 0]
        elif state[-1] == 'a':
            return [0, 0.9, 0.1]
        else:
            assert state[-1] == 'b'
            return [0.9, 0, 0.1]

    def state_char_prob(self, state, token):
        idx = self.internal_alphabet.index(token)
        return self.state_probs_dist(state)[idx]


class CopyTransformerAuto:
    def __init__(self, params, config, tok_to_idx, max_length=2, seed=None) -> None:
        self.params = params
        self.config = config
        self.tok_to_idx = tok_to_idx
        self.max_length = max_length

        if seed == None:
            seed = int(np.random.random() * 1e5)
        
        self.rng = jax.random.PRNGKey(seed)

        self.model = TransformerLM(self.config)
        self.n_symbols = len(tok_to_idx) - 3
        
        self.alpha_symbols = [chr(start_char + i) for i in range(self.n_symbols)]
        symbols = itertools.product(self.alpha_symbols, range(1, self.config.max_item_label + 1))
        self.symbols = [f'{s[0]}|{str(s[1])}' for s in symbols]

        self.all_combos = []
        for i in range(1, self.max_length + 1):
            combo = itertools.product(self.alpha_symbols, repeat=i)
            self.all_combos.extend(combo)
        
        # alphabetical symbols represent true inputs. Numbers represent starting positions
        self.input_alphabet = [''.join(c) for c in self.all_combos] + self.symbols
        self.end_token = 'END'
        self.internal_alphabet = self.input_alphabet + [self.end_token]
        self.terminus_probs = (len(self.internal_alphabet) - 1) * [0] + [1]

        self.c_get_probs = jax.jit(functools.partial(
            get_probs, params=self.params, config=self.config
        ))

    def initial_state(self):
        return ('s',)
    
    def next_state(self, curr_state, token):
        return curr_state + (token,)

    def state_probs_dist(self, state):
        # spread to initial states
        if state[-1] == 's':
            n_combos = len(self.all_combos)
            return n_combos * [1 / n_combos] + (len(self.symbols) + 1) * [0]
        
        assert len(state) >= 2
        try:
            prompt, labels = self._extract_prompt(state)
        except ParseError:
            return self.terminus_probs

        probs = self.c_get_probs(prompt, labels).tolist()
        probs = len(self.all_combos) * [0] + probs
        return probs

    def state_char_prob(self, state, token):
        idx = self.internal_alphabet.index(token)
        return self.state_probs_dist(state)[idx]
    
    def _extract_prompt(self, state):
        beg_prompt = tuple(state[1])
        if '|' in beg_prompt:
            raise ParseError('incorrectly formatted task token')

        beg_labels = np.arange(len(beg_prompt)) + 1

        def split_pair(pair):
            return pair[0], int(pair[1])

        try:
            end_info = [split_pair(s.split('|')) for s in state[2:]]
        except IndexError:
            raise ParseError('input symbol missing divider |')

        if len(end_info) > 0:
            end_prompt, end_labels = zip(*end_info)
        else:
            end_prompt, end_labels = () , ()
        
        prompt = beg_prompt + ('GO',) + end_prompt
        prompt = [self.tok_to_idx[t] for t in prompt]
        labels = np.concatenate((beg_labels, [0], end_labels))
        return jnp.array(prompt), jnp.array(labels, dtype=jnp.int32)


n_symbols = 2
max_item_label = 10

config = TransformerConfig(n_symbols + 3, max_item_label=max_item_label)
train_ds = CopyDataset(range(1, 3+1), vocab_size=n_symbols, max_item_label=max_item_label)
train_dl = to_dataloader(train_ds, batch_size=32, num_workers=0, pin_memory=True)

# <codecell>
state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=3_000, print_every=1_000, save_dir='save/tmp')

# <codecell>
mngr = make_ckpt_manager('save/tmp')
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(best_step, items={
                 'state': None, 'config': TransformerConfig(0)})
params = r['state']['params']

# <codecell>
a = CopyTransformerAuto(params, config, train_ds.tok_to_idx, max_length=2)
# probs = a.state_probs_dist(('s','ab', 'a|1'))
# idx = np.argmax(probs)
# a.internal_alphabet[idx]

# state = a.initial_state()
# tok = None

# while tok != a.end_token:
#     probs = a.state_probs_dist(state)
#     probs = probs / np.sum(probs)
#     tok = np.random.choice(a.internal_alphabet, p=probs)
#     state = a.next_state(state, tok)

# print(state)


p, table, _ = learn(a, 
                pdfas_path='save/pdfa_test', 
                interesting_p_transition_threshold=1e-4,
                s_separating_threshold=-1,
                weight_keep_threshold=1e-4)

# <codecell>
p.draw_nicely(keep=True, filename='tmp', transition_tol=1e-3)

# <codecell>



# <codecell>

# a = DummyAuto()
# p, _, _ = learn(a, pdfas_path='save/pdfa_test')
# p.draw_nicely(keep=True, filename='tmp')


# state = a.initial_state()
# tok = None

# while tok != 'f':
#     probs = a.state_probs_dist(state)
#     tok = np.random.choice(a.internal_alphabet, p=probs)
#     state = a.next_state(state, tok)

# print(state)