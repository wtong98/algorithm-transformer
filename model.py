"""
Adapted from: https://github.com/google/flax/blob/main/examples/lm1b

License notice:
Copyright 2023 The Flax Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# <codecell>
# import os
# os.environ['JAX_PLATFORMS'] = 'cpu'

import functools
import os.path
import shutil
from typing import Callable, Any, Optional

from flax import linen as nn, struct, traverse_util
from flax.core.frozen_dict import freeze, FrozenDict
from flax.training import train_state, orbax_utils
from flax.training.common_utils import stack_forest

import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import optax

from tqdm import tqdm

from task.string_copy import *

@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int
    dtype: Any = jnp.float32
    emb_dim: int = 128
    num_heads: int = 1
    num_layers: int = 6
    qkv_dim: int = 128
    mlp_dim: int = 128
    max_len: int = 100
    decode: bool = False
    kernel_init_name = 'xavier_uniform'
    kernel_init_params: FrozenDict = struct.field(default_factory=FrozenDict)
    bias_init_name = 'normal'
    bias_init_params: FrozenDict = struct.field(default_factory=lambda: FrozenDict({'stddev': 1e-6}))
    posemb_init: Optional[Callable] = None
    posemb_scramble: bool = False
    max_item_label: int = -1  # TODO: unify with max_len
    freeze_embedding: bool = False
    sinus_embedding: bool = False
    nope_embeding: bool = False
    rel_pos_att: bool = False
    rel_pos_rand_max: int = 0


    def kernel_init(self):
        init_f = getattr(nn.initializers, self.kernel_init_name)
        return init_f(**self.kernel_init_params)
    
    def bias_init(self):
        init_f = getattr(nn.initializers, self.bias_init_name)
        return init_f(**self.bias_init_params)


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0,
                    squeeze=False):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
            max_len: maximum possible length for the input.
            min_scale: float: minimum frequency-scale in sine grating.
            max_scale: float: maximum frequency-scale in sine grating.

    Returns:
            output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, :d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)

        if not squeeze:
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]

        return jnp.array(pe)

    return init


class SingleHeadSelfAttention(nn.Module):
    """Single head self attention, with some custom sauce.
    
    Args:
        config: TransformerConfig dataclass with hyperparameters
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, mask=None, idxs=None, use_bias=False):
        dense = functools.partial(
            nn.Dense,
            features=self.config.qkv_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
            use_bias=use_bias)
        
        query = dense(name='query')(inputs)
        key = dense(name='key')(inputs)
        value = dense(name='value')(inputs)
        depth = query.shape[-1]

        if self.config.rel_pos_att:
            attn_weights = RelativePositionAttention(self.config)(query, key, idxs=idxs)
        else:
            attn_weights = jnp.einsum('...qd,...kd->...qk', query, key)

        attn_weights /= jnp.sqrt(depth)

        if mask is not None:
            attn_weights = jnp.where(mask.squeeze(), attn_weights, np.info(np.int32).min)

        attn_weights = jax.nn.softmax(attn_weights)
        self.sow('intermediates', 'attention_weights', attn_weights)

        attn_out = attn_weights @ value
        return attn_out


class RelativePositionAttention(nn.Module):

    config: TransformerConfig

    @nn.compact
    def __call__(self, query, key, idxs=None):
        batch_size, length, depth = query.shape

        content_bias = self.param('content_bias', self.config.kernel_init(), (1, 1, depth))
        content_att = jnp.einsum('bqd,bkd->bqk', query + content_bias, key)

        pe = sinusoidal_init(
            max_len=self.config.max_len,
            squeeze=True)(None, (depth,))
        
        if idxs is not None:
            pe = pe[idxs, :]
        else:
            pe = pe[:length, :]
        
        self.sow('intermediates', 'rand_idxs', idxs)
        pe = jnp.broadcast_to(pe, (batch_size,) + pe.shape)
        pe = jnp.flip(pe)   # relative ordering

        relative_key = nn.Dense(depth, use_bias=False)(pe)
        relative_bias = self.param('relative_bias', self.config.kernel_init(), (1, 1, depth))
        relative_att = jnp.einsum('bqd,bkd->bqk', query + relative_bias, relative_key)

        relative_att = relative_shift(relative_att)

        assert content_att.shape == relative_att.shape
        return content_att + relative_att


# TODO: cite
def relative_shift(x: jax.Array):
    def rel_shift_causal(logits: jax.Array) -> jax.Array:
        """Shifts the relative logits, assuming causal attention.

        Given inputs:
            [[-4, -3, -2, -1],
            [-4, -3, -2, -1]]

        The shifted (and, later, masked) output is:
            [[-3, -2, -1,  0],
            [-4, -3, -2, -1]]

        Args:
            logits: input tensor of shape [T_q, T_v]

        Returns:
            A shifted version of the input of size [T_q, T_v].
        """
        t1, t2 = logits.shape
        # We prepend zeros on the final timescale dimension.
        to_pad = jnp.zeros_like(logits[..., :1])
        x = jnp.concatenate((to_pad, logits), axis=-1)

        # Reshape trick to  shift input.
        x = jnp.reshape(x, [t2 + 1, t1])

        # Remove extra time dimension and re-shape.
        x = jax.lax.slice(x, [1] + [0] * (x.ndim - 1), x.shape)

        return jnp.reshape(x, [t1, t2])
    
    return jax.vmap(rel_shift_causal)(x)


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
            inputs: input data.

        Returns:
            output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                 ' but it is: %d' % inputs.ndim)
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                                    pos_emb_shape,
                                                                    None)
        else:
            pos_embedding = self.param('pos_embedding', config.posemb_init, pos_emb_shape)
        
        if config.posemb_scramble:
            key = self.make_rng('rng')
            rand_idxs = jax.random.choice(key, config.max_len, shape=(length,), replace=False)
            rand_idxs = jnp.sort(rand_idxs)
            pe = pos_embedding[:, rand_idxs, :]
        else:
            pe = pos_embedding[:, :length, :]
        
        if config.nope_embeding or config.rel_pos_att:
            return inputs
        else:
            return inputs + pe


class AddLabelItemEmbs(nn.Module):

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, labels) -> Any:

        initzr = nn.initializers.normal(stddev=1.0)
        if self.config.sinus_embedding:
            initzr = sinusoidal_init(self.config.max_len + 1, squeeze=True)

        emb = nn.Embed(
            num_embeddings=self.config.max_item_label + 1,
            features=self.config.emb_dim,
            embedding_init=initzr
        )(labels)

        return inputs + emb


class EncoderDecoder1DBlock(nn.Module):
    """Transformer encoder-decoder layer.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                decoder_mask=None,
                idxs=None):
        """Applies EncoderDecoder1DBlock module.

        Args:
            inputs: input data for decoder
            decoder_mask: decoder self-attention mask.
            encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
            output after transformer encoder-decoder block.
        """
        config = self.config

        # Decoder block.
        assert inputs.ndim == 3
        x = inputs
        self.sow('intermediates', 'pre_attention', x)
        self.sow('intermediates', 'mask', decoder_mask)
        x = SingleHeadSelfAttention(config)(x, decoder_mask, idxs=idxs)

        x = x + inputs

        return x


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
        shared_embedding: a shared embedding layer to use.
    """
    config: TransformerConfig
    shared_embedding: Any = None

    @nn.compact
    def __call__(self,
                inputs,
                labels=None,
                decoder_mask=None):
        """Applies Transformer model on the inputs.

        Args:
            inputs: input data.
            decoder_mask: decoder self-attention mask.

        Returns:
            output of a transformer decoder.
        """
        config = self.config
        assert inputs.ndim == 2  # (batch, len)

        # Target Embedding
        y = nn.Embed(
                num_embeddings=config.vocab_size,
                features=config.emb_dim,
                embedding_init=nn.initializers.normal(stddev=1.0))(inputs)

        if config.max_item_label > 0:
            y = AddLabelItemEmbs(config=config)(y, labels)
        else:
            y = AddPositionEmbs(config=config)(y)
        y = y.astype(config.dtype)

        rand_idxs = None
        if self.config.rel_pos_rand_max > 0:
            key = self.make_rng('rng')
            rand_idxs = 1 + jax.random.choice(key, self.config.rel_pos_rand_max, shape=(y.shape[1] - 1,), replace=False)
            rand_idxs = jnp.sort(rand_idxs)
            rand_idxs = jnp.concatenate((jnp.zeros(1,), rand_idxs)).astype(int)

        # Target-Input Decoder
        for lyr in range(config.num_layers):
            y = EncoderDecoder1DBlock(
                config=config, name=f'TransformerBlock_{lyr}')(
                        y,
                        decoder_mask=decoder_mask,
                        idxs=rand_idxs)

        logits = nn.Dense(
            config.vocab_size + config.max_item_label + 1,
            dtype=config.dtype,
            kernel_init=config.kernel_init(),
            bias_init=config.bias_init(),
            name='LogitDense')(y)
        return logits


class TransformerLM(nn.Module):
    """Transformer pure decoder stack for language modelling.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                labels=None):
        """Applies TransformerLM on the inputs.

        Args:
            inputs: target data.
            inputs_positions: input subsequence positions for packed examples.
            inputs_segmentation: input segmentation info for packed examples.

        Returns:
            logits array from transformer decoder.
        """
        config = self.config

        # Make padding attention masks.
        if config.decode:
            # for fast autoregressive decoding we use no decoder mask
            decoder_mask = None
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
                nn.make_causal_mask(inputs, dtype=config.dtype))

        logits = Decoder(
            config=config, shared_embedding=None, name='Decoder')(
                    inputs,
                    labels=labels,
                    decoder_mask=decoder_mask)
        return logits.astype(self.config.dtype)

def make_ckpt_manager(save_dir):
    return CheckpointManager(
        save_dir, 
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(
                keep_period=1,
                best_fn=lambda x: x,
                best_mode='min')
        
    )

def train(config: TransformerConfig, train_dl, eval_dl=None, eval_iters=1_000, lr=5e-5, n_iters=10_000, seed=None, print_every=1_000, save_dir='save/model'):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    train_iter = iter(train_dl)
    mngr = make_ckpt_manager(save_dir)

    if seed == None:
        seed = int(np.random.random() * 1e5)

    rng = jax.random.PRNGKey(seed)
    rng, params_rng, global_rng = jax.random.split(rng, num=3)

    max_len = config.max_len
    if config.max_item_label > 0 or config.rel_pos_rand_max > 0:
        max_len = max(config.max_item_label, config.rel_pos_rand_max)

    input_shape = (train_dl.batch_size, max_len)
    model = TransformerLM(config)

    init_var = jax.jit(model.init)({'rng': global_rng, 'params': params_rng}, jnp.ones(input_shape, jnp.float32), labels=jnp.ones(input_shape, jnp.int32))

    opt = optax.adamw(lr)

    if config.freeze_embedding:
        partition = freeze(traverse_util.path_aware_map(
            lambda path, _: 'frozen' if 'AddLabelItemEmbs_0' in path else 'trainable', init_var['params']
        ))

        print('FREEZING', partition)
        optimizer = optax.multi_transform(
            {'trainable': opt, 'frozen': optax.set_to_zero()}, partition
        )
    else:
        optimizer = opt

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_var['params'],
        tx=optimizer
    )
    del init_var

    rng, model_rng = jax.random.split(rng)
    train_metrics = []
    eval_metrics = []

    for i in range(n_iters):
        batch = next(train_iter)
        state, metrics = train_step(state, batch, config, rng=model_rng)
        train_metrics.append(metrics)

        if i % print_every == 0 or i == (n_iters-1):
            if eval_dl != None:
                curr_eval_metrics = []
                for _, batch in zip(range(eval_iters), eval_dl):
                    rng, eval_rng = jax.random.split(rng)
                    metrics = eval_step(state, batch, config, rng=eval_rng)
                    curr_eval_metrics.append(metrics)
                curr_eval_metrics = stack_forest(curr_eval_metrics)
                curr_eval_metrics = jax.tree_util.tree_map(jnp.mean, curr_eval_metrics)
                eval_metrics.append(curr_eval_metrics)
                print_metric(i, curr_eval_metrics, is_eval=True)

                ckpt = {'state': state, 'config': config}
                
                save_args = orbax_utils.save_args_from_target(ckpt)
                mngr.save(i, ckpt, metrics=curr_eval_metrics['loss'].item(), save_kwargs={'save_args': save_args})
            else:
                print_metric(i, train_metrics[-1])
    
    return state, {
        'train_metrics': train_metrics, 
        'eval_metrics': eval_metrics,
        'manager': mngr
    }

def print_metric(step, m, is_eval=False):
    prefix = 'TRAIN'
    if is_eval:
        prefix = 'EVAL'

    print(f'{prefix} step {step}: loss: {m["loss"]:.4f}  acc: {m["accuracy"]:.4f}  aon: {m["aon_accuracy"]:.4f}  conf: {m["confidence"]:.4f}')


@functools.partial(jax.jit, static_argnames='config')
def train_step(state, batch, config, rng=None):
    train_keys = ['inputs', 'labels', 'mask']
    inputs, labels, mask = [batch.get(k, None) for k in train_keys]
    # print('LABS', labels)
    
    rng = jax.random.fold_in(rng, state.step)
    rng, global_rng = jax.random.split(rng)

    def loss_fn(params):
        logits = TransformerLM(config).apply(
            {'params': params},
            inputs,
            labels=labels,
            rngs={'rng': global_rng}
        )
        
        tok_logits, label_logits = logits[...,:config.vocab_size], logits[...,config.vocab_size:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            tok_logits[...,:-1,:], inputs[..., 1:])
        
        if config.max_item_label > 0:
            label_loss = optax.softmax_cross_entropy_with_integer_labels(
                label_logits[...,:-1,:], labels[...,1:]
            )
            loss += label_loss

        loss = loss * mask[...,:-1]

        return loss.sum(axis=1).mean(), logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, inputs, mask, vocab_size=config.vocab_size)
    return new_state, metrics


@functools.partial(jax.jit, static_argnames='config')
def eval_step(state, batch, config, rng=None):
    train_keys = ['inputs', 'labels', 'mask']
    inputs, labels, mask = [batch.get(k, None) for k in train_keys]

    logits = TransformerLM(config).apply({'params': state.params}, inputs, labels=labels, rngs={'rng': rng})
    return compute_metrics(logits, inputs, mask, vocab_size=config.vocab_size)


def predict(prompt, params, config, seed=None, use_tqdm=False):
    if config.max_item_label > 0:
        return predict_with_lab(prompt, params, config, seed=seed, use_tqdm=use_tqdm)
    else:
        return predict_no_lab(prompt, params, config, seed=seed, use_tqdm=use_tqdm)


# TODO: test
def predict_no_lab(prompt, params, config, seed=None, use_tqdm=False):
    prompt = jnp.array(prompt)
    assert len(prompt.shape) == 1
    prompt = prompt.reshape(1, -1)

    if seed == None:
        seed = np.random.randint(1, 99999)
    rng = jax.random.PRNGKey(seed)

    it = range(config.max_len - prompt.shape[1])
    if use_tqdm:
        it = tqdm(it)

    for _ in it:
        prompt, logits = get_next(prompt, params, config, rng=rng)

        if prompt[0,-1] == 2: # END == 2
            break

    return prompt.flatten(), {'logits': logits}


@functools.partial(jax.jit, static_argnames='config')
def get_next(prompt, params, config, rng=None):
    m = TransformerLM(config)
    logits = m.apply({'params': params}, prompt, rngs={'rng': rng})
    nxt_tok = jnp.argmax(logits, -1)[0,-1].reshape(1, 1)

    prompt = jnp.append(prompt, nxt_tok, axis=1)
    return prompt, logits


def predict_with_lab(prompt: list, params: dict, config: TransformerConfig, seed=None, use_tqdm=False):
    prompt = jnp.array(prompt)
    assert len(prompt.shape) == 1
    labels = np.sort(np.random.choice(np.arange(1, config.max_item_label + 1), size=len(prompt) - 1, replace=False))
    labels = np.append(labels, [0])

    prompt = prompt.reshape(1, -1)
    labels = labels.reshape(1, -1)

    if seed == None:
        seed = np.random.randint(1, 99999)
    rng = jax.random.PRNGKey(seed)

    it = range(config.max_len - prompt.shape[1])
    if use_tqdm:
        it = tqdm(it)

    for _ in it:
        prompt, labels = get_next_with_lab(prompt, labels, params, config, rng=rng)

        if prompt[0,-1] == 2: # END == 2
            break
    
    return prompt.flatten(), {'labels': labels}


@functools.partial(jax.jit, static_argnames='config')
def get_next_with_lab(prompt, labels, params, config, rng=None):
    m = TransformerLM(config)
    logits = m.apply({'params': params}, prompt, labels=labels, rngs={'rng': rng})
    logits_tok, logits_lab = logits[...,:config.vocab_size], logits[...,config.vocab_size:]

    nxt_tok = jnp.argmax(logits_tok, -1)[0,-1].reshape(1, 1)
    nxt_lab = jnp.argmax(logits_lab, -1)[0,-1].reshape(1, 1)

    prompt = jnp.append(prompt, nxt_tok, axis=1)
    labels = jnp.append(labels, nxt_lab, axis=1)
    return prompt, labels


def compute_metrics(logits, inputs, mask, vocab_size=None):
    if vocab_size == None:
        vocab_size = logits.shape[-1]

    pred_tok_logits = logits[...,:-1,:vocab_size]
    pred_inputs = inputs[...,1:]
    pred_mask = mask[...,:-1]

    loss = optax.softmax_cross_entropy_with_integer_labels(pred_tok_logits, pred_inputs)
    loss = loss * pred_mask
    loss = loss.sum(axis=1).mean()

    preds = jnp.argmax(pred_tok_logits, axis=-1)
    acc = jnp.sum((preds == pred_inputs) * pred_mask) / jnp.sum(pred_mask)

    aon_acc = jnp.sum((preds == pred_inputs) * pred_mask, axis=1) / jnp.sum(pred_mask, axis=1)
    aon_acc = jnp.mean(jnp.isclose(aon_acc, 1))

    probs = jax.nn.softmax(pred_tok_logits)[...,pred_inputs]
    probs = jnp.diagonal(probs, axis1=1, axis2=3)
    probs = jnp.diagonal(probs, axis1=0, axis2=1).T
    conf = jnp.sum(probs * pred_mask) / jnp.sum(pred_mask)

    return {
        'loss': loss,
        'accuracy': acc,
        'aon_accuracy': aon_acc,
        'confidence': conf
    }


def evaluate_acc(length, params, config, max_item_label=-1, n_symbols=2, n_examples=100, use_tqdm=False):
    train_ds = CopyDataset(length, vocab_size=n_symbols,
                           max_item_label=max_item_label)

    n_correct = 0
    fails = []

    it = zip(range(n_examples), iter(train_ds))
    if use_tqdm:
        it = tqdm(it, total=n_examples)

    for _, example in it:
        ans = example[0]
        prompt = ans[:len(ans)//2]
        try:
            pred = predict(prompt, params, config)
        except Exception as e:
            print('failed to predict: ', e)
            fails.append((prompt, None))
            continue

        if hasattr(pred, '__len__'):
            pred = pred[0]

        if pred.shape == ans.shape and np.all(pred == ans):
            n_correct += 1
        else:
            fails.append((prompt, pred))

    return n_correct / n_examples, fails
# <codecell>
# '''
n_symbols = 2
max_item_label = 50
max_train_len = 10


# config = TransformerConfig(
#     n_symbols + 3, max_item_label=max_item_label)
# train_ds = CopyDataset(range(1, 10+1), vocab_size=n_symbols,
#                        max_item_label=max_item_label)

config = TransformerConfig(
    n_symbols + 3, rel_pos_att=True, rel_pos_rand_max=max_item_label*2, max_len=512)
    # n_symbols + 3, max_item_label=max_item_label//2, max_len=512)
train_ds = CopyDataset(range(1, max_train_len+1), vocab_size=n_symbols, max_item_label=max_item_label//2)

train_dl = to_dataloader(train_ds, batch_size=32,
                         num_workers=0, pin_memory=True)

# <codecell>
state, info = train(config, train_dl, eval_dl=train_dl,
                    n_iters=20_000, print_every=1_000, save_dir='scratch/save/tmp')

# <codecell>
train = stack_forest(info['train_metrics'])
test = stack_forest(info['eval_metrics'])

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

for ax, metrics in zip(axs, [train, test]):
    ax.plot(metrics['accuracy'], color='C0', label='accuracy', alpha=0.8)
    ax.set_ylabel('Accuracy', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')

    ax.plot(metrics['aon_accuracy'], color='C0', label='aon_accuracy', alpha=0.6, linestyle='dashed')
    ax.set_xscale('log')

    ax2 = ax.twinx()
    ax2.plot(metrics['loss'], color='C1', label='loss', alpha=0.8)
    ax2.set_ylabel('Loss', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    # ax.plot(metrics['confidence'], label='confidence')
    # ax.plot(metrics['loss'], label='loss')

# plt.savefig('scratch/fig/item_label_loss_curve.png')

# <codecell>
mngr = make_ckpt_manager('scratch/save/tmp')
best_step = mngr.best_step()
print('BEST ITER', best_step)

r = mngr.restore(mngr.latest_step())
raw_state = r['state']

# %%
inputs = [4, 3, 3, 4, 3, 3, 1]
# predict_with_lab(inputs, raw_state['params'], config)
seq, info = predict(inputs, raw_state['params'], config)
seq

# m = TransformerLM(config)
# _, intm = m.apply({'params': raw_state['params']}, jnp.array(inputs).reshape(1, -1), mutable='intermediates', rngs={'rng': jax.random.PRNGKey(5)})
# intm['intermediates']['Decoder']['TransformerBlock_0']['SingleHeadSelfAttention_0']['RelativePositionAttention_0']['rand_idxs'][0]

# <codecell>
evaluate_acc(11, raw_state['params'], config, max_item_label=max_item_label, n_examples=10)

# %%

def get_attn_weights(seq, params, config, labels=None):
    all_weights = []
    if labels is not None:
        labels = labels.reshape(1, -1)

    for i in range(config.num_layers):
        m = TransformerLM(config)
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
    seq, labs = predict(jnp.array(in_seq), params, config)
    # seq = jnp.array([3,3,4,3,4,1,3,3,4,3,4])
    seq = seq[:(len(in_seq)*2)]
    print('SEQ', seq)
    attn_weights = get_attn_weights(seq, params, config, labels=labs)
    plot_attn_weights(attn_weights, seq, train_ds.idx_to_tok)


# plot_sequence([3,3,4,3,4,4,4,3,3,4,3,4,3,3,4,3,3,4,1], raw_state['params'], pred_config)
plot_sequence(inputs, raw_state['params'], config)
# plt.savefig('fig/fix_sinus_attn_15.png')

# %%
'''