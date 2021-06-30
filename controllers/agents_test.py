import jax
import jax.numpy as jnp
import random
import unittest

from agents import RLParams, DQNPureJax

class TestDQNPureJax(unittest.TestCase):
  def test_action(self):
    params = RLParams(nn_sizes=[20, 100, 100, 3],
                      gamma=0.9,
                      lr=0.01)
    # NN
    n = DQNPureJax(params)
    # Random key.
    key = jax.random.PRNGKey(random.randint(0, 88))
    # Run FV.
    fv = jax.random.normal(key, (20,))
    # Get action
    action = n.Action(fv)

    self.assertTrue(action in (0, 1, 2))

  def test_train(self):
    params = RLParams(nn_sizes=[2, 5, 5, 3],
                      gamma=0.9,
                      lr=0.00001)
    # NN
    n = DQNPureJax(params)

    # Train input.
    fv = jnp.array([1.0, 2.0])
    next_fv = jnp.array([3.0, 4.0])
    action = 1
    reward = 8.0

    loss1 = n.TrainStep(fv, action, reward, next_fv)
    # Train another 2 steps with the same data frame.
    loss2 = n.TrainStep(fv, action, reward, next_fv)
    loss3 = n.TrainStep(fv, action, reward, next_fv)

    # Loss should keep getting reduced.
    self.assertTrue(loss2 < loss1)
    self.assertTrue(loss3 < loss2)


if __name__ == '__main__':
  unittest.main()
