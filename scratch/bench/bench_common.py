"""
Common utilities for benchmarking stuff
"""
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.append('../../')

from model import *
from task.string_copy import *


@dataclass
class Case:
    name: str
    config: TransformerConfig
    save_dir: str
    train_iters: int = 30_000
    res: dict = field(default_factory=dict)
    ds_kwargs: dict = field(default_factory=dict)
    fine_tune_split: float | None = None


def evaluate_acc(length, params, config, n_examples=100, use_tqdm=False, go_tok=1, end_tok=2):
    kwargs = config.ds_generator_kwargs.copy({'lengths': length})
    config = config.replace(ds_generator_kwargs=kwargs, vocab_size=params['Embed_0']['embedding'].shape[0])
    train_ds, config = CopyDataset.from_config(config, unify_config=False)

    n_correct = 0

    it = zip(range(n_examples), iter(train_ds))
    if use_tqdm:
        it = tqdm(it, total=n_examples)

    for _, example in it:
        ans = example[0]
        offset = 1 if train_ds.bos else 0
        prompt = ans[:len(ans)//2+offset]

        try:
            pred = predict(prompt, params, config, go_tok=go_tok, end_tok=end_tok)

        except Exception as e:
            print('failed to predict: ', e)
            continue

        if hasattr(pred, '__len__'):
            pred = pred[0]

        if pred.shape == ans.shape and np.all(pred == ans):
            n_correct += 1

    return n_correct / n_examples


def run_train(all_cases, skip_existing=False, batch_size=32):
    for case in all_cases:
        if skip_existing and Path(case.save_dir).exists():
            print('SKIPPING', case.name)
            continue

        print('TRAINING', case.name)

        init_params = None
        if case.fine_tune_split is not None:
            print('(training base)')
            train_ds, case.config = GenerativeDataset.from_config(case.config)
            train_dl = to_dataloader(train_ds, batch_size=batch_size, pin_memory=True)

            n_iters = int(case.fine_tune_split * case.train_iters)
            state, info = train(case.config, train_dl, eval_dl=train_dl, n_iters=n_iters, print_every=1000)
            init_params = state.params

        train_ds, case.config = CopyDataset.from_config(case.config)
        train_dl = to_dataloader(train_ds, batch_size=batch_size,
                                num_workers=0, pin_memory=True)

        _, info = train(case.config, train_dl, init_params=init_params, eval_dl=train_dl,
                        n_iters=case.train_iters, print_every=1_000, save_dir=case.save_dir)
        plot_train_metrics(info, save_path=case.save_dir + '/metrics.png')
        plt.close()
