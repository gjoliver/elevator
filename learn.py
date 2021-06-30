from collections import deque
import ray

from simulator.evolver import EvolverConfig, Evolver

ITERATION = 10


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
      evolver.step()
      episode.append(evolver.state())

    # Synchronously add the episode.
    # Not really necessary, but why not.
    ray.wait([self._rb.AddEpisode.remote(episode)])
    self._count += 1


@ray.remote
class ReplayBuffer(object):
  def __init__(self, size=100):
    self._queue = deque(maxlen=size)

  def AddEpisode(self, episode):
    self._queue.append(episode)

  def Size(self):
    return len(self._queue)


def main():
  ray.init()

  rb = ReplayBuffer.remote()
  simulators = [Simulator.remote(i, rb) for i in range(2)]

  for _ in range(ITERATION):
    sims = [s.Simulate.remote() for s in simulators]
    ray.wait(sims, num_returns=len(sims))

    num_episodes = rb.Size.remote()
    print(ray.get(num_episodes))


if __name__ == '__main__':
  main()
