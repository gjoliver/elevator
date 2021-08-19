from collections import deque
import jax.numpy as jnp
import random
import ray

from controllers.agents import DQNPureJax
from controllers.rl import FV, NumFeatures, RLAssigner
from simulator.evolver import EvolverConfig, Evolver
from utils.hyper_params import HyperParams


@ray.remote
class Simulator(object):
  def __init__(self, replica, rb, cfg):
    self._replica = replica
    self._count = 0
    self._rb = rb
    self._cfg = cfg

  def UpdateController(self, agent_id):
    agent = ray.get(agent_id)
    # Update controller, which will get used in next Simulate() call.
    self._cfg.controller = RLAssigner(self._cfg.hparams, agent)

  def Simulate(self):
    episode = []
    evolver = Evolver(self._cfg)
    while evolver.time() < self._cfg.horizon:
      rider, state, action, reward = evolver.step()
      episode.append([rider, state, action, reward])

    # Synchronously add the episode.
    # Not really necessary once we make all the jobs work asynchronously.
    ray.wait([self._rb.AddEpisode.remote(episode)])
    self._count += 1


@ray.remote
class ReplayBuffer(object):
  def __init__(self, hparams):
    # Small replay buffer. Same size as batch_size.
    # Basically doing on-policy training for now.
    self._queue = deque(maxlen=hparams.batch_size)
    self._hparams = hparams

  def AddEpisode(self, episode):
    # Compute reward + future discounted reward.
    f_reward = 0
    for i in reversed(range(len(episode))):
      _, _, _, reward = episode[i]
      d_reward = reward + self._hparams.gamma * f_reward
      episode[i][3] = d_reward
      f_reward = d_reward

    # This is kind of nuts. We hardcode to randomly choose half of the
    # batch size for now, assuming that there are only 2 simulators.
    # TODO(jungong) : fix it.
    selected = random.choices(
      range(len(episode) - 1), k=int(self._hparams.batch_size / 2))
    for i in selected:
      rider, state, action, reward = episode[i]
      fv = FV(state, rider, self._hparams.num_floors)
      next_rider, next_state, _, _ = episode[i + 1]
      next_fv = FV(next_state, next_rider, self._hparams.num_floors)
      # Add 1 frame.
      self._queue.append([fv, action, reward, next_fv])

  def GetFrames(self):
    fvs = jnp.vstack([fv for fv, _, _, _ in self._queue])
    actions = jnp.array([action for _, action, _, _ in self._queue])
    rewards = jnp.array([reward for _, _, reward, _ in self._queue])
    next_fvs = jnp.vstack([next_fv for _, _, _, next_fv in self._queue])
    return [fvs, actions, rewards, next_fvs]

  def Size(self):
    return len(self._queue)


@ray.remote
class Trainer(object):
  def __init__(self, hparams, rb):
    self._rb = rb
    self._agent = DQNPureJax(hparams)

  def SaveAgent(self):
    return ray.put(self._agent)

  def Step(self):
    return self._agent.TrainStep(*ray.get(self._rb.GetFrames.remote()))


@ray.remote
def Eval(hparams, agent_id):
  cfg = EvolverConfig(hparams=hparams,
                      horizon=1000,  # Eval horizon.
                      controller=RLAssigner(hparams))
  e = Evolver(cfg)
  while e.time() < cfg.horizon:
    e.step()

  # Print intermediate eval results.
  print(e.stats())


def main():
  ray.init()

  # Note, widths of the input/output layers should match the
  # number of elevators and floors.
  hparams = HyperParams(nn_sizes=[NumFeatures(6, 4), 100, 100, 4],
                        num_elevators=4,
                        num_floors=6)

  rb = ReplayBuffer.remote(hparams)

  evolver_cfg = EvolverConfig(hparams=hparams,
                              horizon=300,
                              controller=RLAssigner(hparams))
  simulators = [Simulator.remote(i, rb, cfg=evolver_cfg)
                for i in range(2)]

  trainer = Trainer.remote(hparams, rb)

  for i in range(hparams.training_iterations):
    sims = [s.Simulate.remote() for s in simulators]
    # TODO(jungong), Sims should be asynchronous.
    ray.wait(sims, num_returns=len(sims))

    # One training step.
    loss = ray.get([trainer.Step.remote()])

    # Broadcast current agent NN to all the simulators.
    agent_id = trainer.SaveAgent.remote()
    ray.wait([s.UpdateController.remote(agent_id)
              for s in simulators])

    # Eval after each iteration.
    Eval.remote(hparams, agent_id)

    print(f'iteration {i}, loss {loss}')


if __name__ == '__main__':
  main()
