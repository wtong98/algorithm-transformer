"""
Experiment with PDFA extraction

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np

import sys
sys.path.append('../')


class DummyAuto:
    def __init__(self) -> None:
        self.input_alphabet = ['a', 'b']
        self.end_token = ['f']
        self.internal_alphabet = self.input_alphabet + self.end_token

    def initial_state(self):
        return ['s']

    def next_state(self, curr_state, token):
        return curr_state + [token]

    def state_probs_dist(self, state):
        if state[-1] == 'a':
            return [0, 0.9, 0.1]
        else:
            return [0.9, 0, 0.1]

    def state_char_prob(self, state, token):
        idx = self.internal_alphabet.index(token)
        return self.state_probs_dist(state)[idx]

a = DummyAuto()

state = a.initial_state()
tok = None

while tok != 'f':
    probs = a.state_probs_dist(state)
    tok = np.random.choice(a.internal_alphabet, p=probs)
    state = a.next_state(state, tok)

print(state)