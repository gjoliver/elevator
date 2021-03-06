import jax
import jax.numpy as jnp
import random


class DQNPureJax(object):
  def __init__(self, hparams):
    nn_sizes = hparams.nn_sizes

    # Simple sequential.
    keys = jax.random.split(jax.random.PRNGKey(random.randint(0, 10000)),
                            len(nn_sizes))

    def layer(in_dim, out_dim, key):
      w_key, b_key = jax.random.split(key)
      return [jax.random.normal(w_key, (out_dim, in_dim)),
              jax.random.normal(b_key, (out_dim,))]

    self._model = [layer(in_dim, out_dim, key)
                   for in_dim, out_dim, key
                   in zip(nn_sizes[:-1], nn_sizes[1:], keys)]

    self._hparams = hparams

  @staticmethod
  def _forward(model, fv):
    def relu(out):
      return jnp.maximum(0, out)

    x = fv
    for w, b in model[:-1]:
      x = jnp.dot(w, x) + b
      # Hidden layers have Relu activation.
      x = relu(x)

    w, b = model[-1]
    logits = jnp.dot(w, x) + b

    return logits

  def Action(self, fvs):
    action = lambda fv: jnp.argmax(self._forward(self._model, fv))
    return jax.vmap(action)(fvs)

  def TrainStep(self, fvs, actions, rewards, next_fvs):
    gamma = self._hparams.gamma

    def loss_fn(model):
      q_fn = jax.vmap(lambda fv: self._forward(model, fv))
      qs = q_fn(fvs)
      one_hot = jax.vmap(lambda q, action: q[action])
      selected_qs = one_hot(qs, actions)
      max_next_qs = jnp.max(q_fn(next_fvs), axis=1)
      targets = rewards + gamma * max_next_qs
      # MSE loss.
      return jnp.mean(jnp.power((targets - selected_qs), 2))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(self._model)

    # Apply gradient.
    lr = self._hparams.learning_rate
    for l, g in zip(self._model, grad):
      l[0] -= lr * g[0]  # Update W
      l[1] -= lr * g[1]  # Update b

    return loss
