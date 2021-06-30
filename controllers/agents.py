import jax
import jax.numpy as jnp


class RLParams(object):
  def __init__(self, nn_sizes, gamma, lr):
    self.nn_sizes = nn_sizes
    self.gamma = gamma
    self.lr = lr


class DQNPureJax(object):
  def __init__(self, params):
    nn_sizes = params.nn_sizes

    # Simple sequential.
    keys = jax.random.split(jax.random.PRNGKey(0), len(nn_sizes))

    def layer(in_dim, out_dim, key):
      w_key, b_key = jax.random.split(key)
      return [jax.random.normal(w_key, (out_dim, in_dim)),
              jax.random.normal(b_key, (out_dim,))]

    self._model = [layer(in_dim, out_dim, key)
                   for in_dim, out_dim, key
                   in zip(nn_sizes[:-1], nn_sizes[1:], keys)]

    self._params = params

  @staticmethod
  def _forward(model, fv):
    def relu(out):
      return jnp.maximum(0, out)

    x = fv
    for w, b in model[:-1]:
      x = jnp.dot(w, x) + b
      x = relu(x)

    w, b = model[-1]
    logits = jnp.dot(w, x) + b

    return logits

  def Action(self, fv):
    return jnp.argmax(self._forward(self._model, fv))

  def TrainStep(self, fv, action, reward, next_fv):
    gamma = self._params.gamma

    def loss_fn(model):
      q = self._forward(model, fv)[action]
      max_next_q = jnp.max(self._forward(model, next_fv))
      target = reward + gamma * max_next_q
      # MSE loss.
      return jnp.power((target - q), 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(self._model)

    # Apply gradient.
    lr = self._params.lr
    for l, g in zip(self._model, grad):
      l[0] -= lr * g[0]  # Update W
      l[1] -= lr * g[1]  # Update b

    return loss
