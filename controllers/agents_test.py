import jax
import jax.numpy as jnp
import random
import unittest

from agents import AgentParams, DQNPureJax

class TestDQNPureJax(unittest.TestCase):
  def test_action(self):
    params = AgentParams(nn_sizes=[20, 100, 100, 3],
                         gamma=0.9,
                         lr=0.01)
    # NN
    n = DQNPureJax(params)
    # Random key.
    key = jax.random.PRNGKey(random.randint(0, 88))
    # Run FV.
    fvs = jax.random.normal(key, (10, 20))
    # Get action
    actions = n.Action(fvs)

    self.assertEqual(len(actions), 10)
    for action in actions:
      self.assertTrue(action in (0, 1, 2))

  def test_train(self):
    params = AgentParams(nn_sizes=[2, 5, 5, 3],
                         gamma=0.9,
                         lr=0.001)
    # NN
    n = DQNPureJax(params)

    # Random key.
    key = jax.random.PRNGKey(random.randint(0, 88))
    # Train input.
    fvs = jax.random.normal(key, (10, 2))
    next_fvs = jax.random.normal(key, (10, 2))
    actions = jnp.array([random.choice([0, 1, 2]) for _ in range(10)])
    rewards = jnp.array([random.uniform(0, 10.0) for _ in range(10)])

    loss1 = n.TrainStep(fvs, actions, rewards, next_fvs)
    # Train another 2 steps with the same data frames.
    loss2 = n.TrainStep(fvs, actions, rewards, next_fvs)
    loss3 = n.TrainStep(fvs, actions, rewards, next_fvs)

    # Loss should keep getting smaller.
    self.assertTrue(loss2 < loss1)
    self.assertTrue(loss3 < loss2)


if __name__ == '__main__':
  unittest.main()
