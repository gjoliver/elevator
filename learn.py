from collections import deque
import jax.numpy as jnp
import ray

from controllers.agents import AgentParams, DQNPureJax
from controllers.rl import FV, NumFeatures, RLAssigner
from simulator.evolver import EvolverConfig, Evolver

# Hyper-parameters.
# TODO(jungong): make this into a struct
ITERATION = 10
NUM_ELEVATORS = 4
NUM_FLOORS = 6
BATCH_SIZE=128
NN_SIZES = [NumFeatures(NUM_FLOORS, NUM_ELEVATORS), 100, 100, NUM_ELEVATORS]
GAMMA = 0.9
LEARNING_RATE = 0.001


@ray.remote
class Simulator(object):
  def __init__(self, replica, rb, cfg=EvolverConfig()):
    self._replica = replica
    self._count = 0
    self._rb = rb
    self._cfg = cfg

  def Simulate(self):
    episode = []
    evolver = Evolver(self._cfg)
    while evolver.time() < self._cfg.horizon:
      rider, state, action, reward = evolver.step()
      episode.append((rider, state, action, reward))

    # Synchronously add the episode.
    # Not really necessary once we make all the jobs work asynchronously.
    ray.wait([self._rb.AddEpisode.remote(episode)])
    self._count += 1


@ray.remote
class ReplayBuffer(object):
  # Small replay buffer. Basically doing on-policy training for now.
  def __init__(self, size=BATCH_SIZE, gamma=GAMMA, num_floors=NUM_FLOORS):
    self._queue = deque(maxlen=size)
    self._batch_size = size
    self._gamma = gamma
    self._num_floors = num_floors

  def AddEpisode(self, episode):
    # Compute reward + future discounted reward.
    f_reward = 0
    for i in reverse(range(len(episode))):
      _, _, _, reward = episode[i]
      d_reward = reward + self._gamma * f_reward
      episode[i][3] = d_reward
      f_reward = d_reward

    # This is kind of nuts. We hardcode to randomly choose half of the
    # batch size for now, assuming that there are only 2 simulators.
    # TODO(jungong) : fix it.
    selected = random.choices(
      range(len(episode) - 1), k=self._batch_size / 2)
    for i in selected:
      rider, state, action, reward = episode[i]
      fv = FV(state, rider, self._num_floor)
      next_rider, next_state, _, _ = episode[i + 1]
      next_fv = FV(next_state, next_rider, self._num_floor)
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
  def __init__(self, rb):
    self._rb = rb
    agent_params = AgentParams(nn_sizes=NN_SIZES,
                               gamma=GAMMA,
                               lr=LEARNING_RATE)
    self._agent = DQNPureJax(agent_params)

  def Step(self):
    self._agent.TrainStep(*ray.get(self._rb.GetFrames.remote()))


def main():
  ray.init(dashboard_host="127.0.0.1")

  rb = ReplayBuffer.remote()

  evolver_cfg = EvolverConfig(num_elevators=NUM_ELEVATORS,
                              num_floors=NUM_FLOORS,
                              controller=RLAssigner(NUM_ELEVATORS,
                                                    NUM_FLOORS))
  simulators = [Simulator.remote(i, rb, cfg=evolver_cfg)
                for i in range(2)]

  trainer = Trainer.remote(rb)

  for i in range(ITERATION):
    print(f'iteration {i}')

    sims = [s.Simulate.remote() for s in simulators]
    # TODO, these should all be asynchronous.
    ray.wait(sims, num_returns=len(sims))

    trainer.Step.remote()


if __name__ == '__main__':
  main()
