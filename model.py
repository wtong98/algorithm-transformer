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
import functools
from typing import Callable, Any, Optional


from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training.common_utils import stack_forest

import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from task import *


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 128
  num_heads: int = 1
  num_layers: int = 1
  qkv_dim: int = 128
  mlp_dim: int = 512
  max_len: int = 128
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
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
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs,
               inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

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
      pos_embedding = self.param('pos_embedding', config.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = (inputs.shape[-1] if self.out_dim is None
                      else self.out_dim)
    x = nn.Dense(
        config.mlp_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init)(
            x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=config.deterministic)
    return output


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Args:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None):
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
    x = nn.LayerNorm(dtype=config.dtype)(inputs)
    self.sow('intermediates', 'pre_attention', x)
    self.sow('intermediates', 'mask', decoder_mask)
    x = nn.SelfAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        decode=config.decode)(x, decoder_mask)
    self.sow('intermediates', 'post_attention', x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic)
    x = x + inputs

    # MLP block.
    z = nn.LayerNorm(dtype=config.dtype)(x)
    z = MlpBlock(config=config)(z)

    return x + z


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
               inputs_positions=None,
               inputs_segmentation=None,
               decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config
    assert inputs.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=config.vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = inputs.astype('int32')
    y = output_embed(y)
    y = AddPositionEmbs(
        config=config, decode=config.decode, name='PositionEmb')(
            y, inputs_positions=inputs_positions)
    y = nn.Dropout(rate=config.dropout_rate)(
        y, deterministic=config.deterministic)
      
    y = y.astype(config.dtype)

    # Target-Input Decoder
    for lyr in range(config.num_layers):
      y = EncoderDecoder1DBlock(
          config=config, name=f'TransformerBlock_{lyr}')(
              y,
              decoder_mask=decoder_mask)
    y = nn.LayerNorm(dtype=config.dtype, name='FinalNorm')(y)

    # Decoded Logits
    if config.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          config.vocab_size,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name='LogitDense')(
              y)
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
               inputs_positions=None,
               inputs_segmentation=None):
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

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              inputs_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=config.dtype))

    logits = Decoder(
        config=config, shared_embedding=None, name='Decoder')(
            inputs,
            inputs_positions=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            decoder_mask=decoder_mask)
    return logits.astype(self.config.dtype)


def train(config, train_dl, eval_dl=None, eval_iters=1_000, lr=5e-5, n_iters=10_000, seed=51, print_every=1_000):
    eval_config = config.replace(deterministic=True)
    train_iter = iter(train_dl)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    input_shape = (train_dl.batch_size, config.max_len)
    model = TransformerLM(eval_config)
    init_var = model.init(init_rng, jnp.ones(input_shape, jnp.float32))
    # print(jax.tree_util.tree_map(lambda x: x.shape, init_var))

    optimizer = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=init_var['params'],
        tx=optimizer
    )
    del init_var

    rng, dropout_rng = jax.random.split(rng)
    train_metrics = []
    eval_metrics = []

    # compile for training
    c_train_step = jax.jit(
        functools.partial(train_step, config=config)
    )

    c_eval_step = jax.jit(
        functools.partial(eval_step, config=eval_config)
    )

    for i in range(n_iters):
        batch = next(train_iter)
        state, metrics = c_train_step(state, batch, rng=dropout_rng)
        # state, metrics = train_step(state, batch, config, rng=dropout_rng)
        train_metrics.append(metrics)

        if i % print_every == 0 or i == (n_iters-1):
            if eval_dl != None:
                curr_eval_metrics = []
                for _, batch in zip(range(eval_iters), eval_dl):
                    metrics = c_eval_step(state, batch)
                    curr_eval_metrics.append(metrics)
                curr_eval_metrics = stack_forest(curr_eval_metrics)
                curr_eval_metrics = jax.tree_util.tree_map(jnp.mean, curr_eval_metrics)
                eval_metrics.append(curr_eval_metrics)
                print_metric(i, curr_eval_metrics, is_eval=True)
            else:
                print_metric(i, train_metrics[-1])
    
    return state

def print_metric(step, m, is_eval=False):
    if is_eval:
        print(f'EVAL step {step}: loss: {m["loss"]:.4f}  acc: {m["accuracy"]:.4f}  conf: {m["confidence"]:.4f}')
    else:
        print(f'TRAIN step {step}: loss: {m["loss"]:.4f}  acc: {m["accuracy"]:.4f}  conf: {m["confidence"]:.4f}')


def train_step(state, batch, config, rng=None):
    train_keys = ['inputs', 'inputs_segmentation',
                  'inputs_position', 'mask']
    inputs, inputs_positions, inputs_segmentation, mask = [
        batch.get(k, None) for k in train_keys]
    
    # weights = jnp.where(inputs > 0, 1., 0.)  # TODO: confirm works correctly with mixed batches
    # pred_idxs = (jnp.sum(weights, axis=1) - 1).astype(jnp.int32)
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        logits = TransformerLM(config).apply(
            {'params': params},
            inputs,
            # inputs_positions=inputs_positions.reshape(1, -1),
            # inputs_segmentation=inputs_segmentation.reshape(1, -1),
            rngs={'dropout': dropout_rng}
        )
        # print('LOGITS', logits.shape)
        # print('INPUTS', inputs.shape)
        # print('RAW INP', inputs)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[...,:-1,:], inputs[..., 1:])
        
        # print('LOSS', loss.shape)
        # print('RAW LOSS BEF', loss)
        # print('MASK', mask[...,:-1])
        loss = loss * mask[...,:-1]
        # print('RAW LOSS', loss.sum(axis=1))
        return loss.sum(axis=1).mean(), logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, inputs, mask)
    return new_state, metrics


def eval_step(state, batch, config):
    inputs = batch['inputs']
    mask = batch['mask']

    logits = TransformerLM(config).apply({'params': state.params}, inputs)
    return compute_metrics(logits, inputs, mask)


def predict(state, prompt, config, eos_id):
    assert len(prompt.shape) == 1
    prompt = prompt.reshape(1, -1)

    m = TransformerLM(config)
    for _ in range(config.max_len - len(prompt)):
        logits = m.apply({'params': state.params}, prompt)
        nxt_tok = jnp.argmax(logits, -1)[0,-1].reshape(1, 1)
        prompt = jnp.append(prompt, nxt_tok, axis=1)

        if nxt_tok.item() == eos_id:
            break

    return prompt.flatten()


@jax.jit
def compute_metrics(logits, inputs, mask):
    pred_logits = logits[...,:-1,:]
    pred_inputs = inputs[...,1:]
    pred_mask = mask[...,:-1]

    loss = optax.softmax_cross_entropy_with_integer_labels(pred_logits, pred_inputs)
    loss = loss * pred_mask
    loss = loss.sum(axis=1).mean()

    preds = jnp.argmax(pred_logits, axis=-1)
    acc = jnp.sum((preds == pred_inputs) * pred_mask) / jnp.sum(pred_mask)
    # print('ACC', acc)

    # probs = jax.nn.softmax(logits)[np.arange(len(logits)), inputs]
    # print('LOGITS', jax.nn.softmax(pred_logits))
    # print('INPUTS', pred_inputs)
    probs = jax.nn.softmax(pred_logits)[...,pred_inputs]
    probs = jnp.diagonal(probs, axis1=1, axis2=3)
    probs = jnp.diagonal(probs, axis1=0, axis2=1).T
    # print('RAW PR', probs * pred_mask)
    conf = jnp.sum(probs * pred_mask) / jnp.sum(pred_mask)

    return {
        'loss': loss,
        'accuracy': acc,
        'confidence': conf
    }

config = TransformerConfig(5)
train_ds = CopyDataset([1, 2, 3, 4])
train_dl = to_dataloader(train_ds, batch_size=32, num_workers=0, pin_memory=True)

# <codecell>
state = train(config, train_dl, eval_dl=train_dl, n_iters=5000, print_every=500)

# %%
pred_config = config.replace(deterministic=True)
predict(state, jnp.array([3,3,4,1]), pred_config, train_ds.tok_to_idx['END'])

# %%
m = TransformerLM(pred_config)
_, intm = m.apply({'params': state.params}, jnp.array([3,3,4,1,3,3,4]).reshape(1, -1), mutable='intermediates')
x = intm['intermediates']['Decoder']['TransformerBlock_0']['pre_attention'][0]
x_out = intm['intermediates']['Decoder']['TransformerBlock_0']['post_attention'][0]
mask = intm['intermediates']['Decoder']['TransformerBlock_0']['mask'][0]

att = state.params['Decoder']['TransformerBlock_0']['SelfAttention_0']
wq = att['query']['kernel']
wk = att['key']['kernel']
wv = att['value']['kernel']
w_out = att['out']['kernel']

# jax.tree_map(lambda x: x.shape, state.params)
x.shape
wq.shape

query = jnp.einsum('...lf,fhd->...lhd', x, wq)
key = jnp.einsum('...lf,fhd->...lhd', x, wk)
value = jnp.einsum('...lf,fhd->...lhd', x, wv)

depth = query.shape[-1]
query /= jnp.sqrt(depth)
attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

attn_weights = jnp.where(mask, attn_weights, -99999)
attn_weights = jax.nn.softmax(attn_weights)
attn_out = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

attn_out = jnp.einsum('...lhd,hdf->...lf', attn_out, w_out)
attn_weights.shape
# jnp.sum(attn_out == x_out)

plt.imshow(attn_weights[0,0])
plt.gca().set_xticklabels(['0', 'a', 'a', 'b', 'GO', 'a', 'a', 'b'])
plt.gca().set_yticklabels(['0', 'a', 'a', 'b', 'GO', 'a', 'a', 'b'])
plt.savefig('fig/tmp_attention.png')
# attn_weights.shape





# %%
