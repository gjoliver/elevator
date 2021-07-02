from controllers.agents import DQNPureJax
import jax
import jax.numpy as jnp


def FVElevator(e, num_floors):
  running_fv = jax.nn.one_hot(e['running'].value - 1, 3)
  floor_fv = jax.nn.one_hot(e['floor'], num_floors)
  stops_fv = [0.0 for _ in range(num_floors)]
  for s in e['stops']:
    # s may include 2 stops, pickup, then dropoff floors.
    for sp in s:
      stops_fv[sp] = 1.0
  stops_fv = jnp.array(stops_fv)
  return jnp.hstack([running_fv, floor_fv, stops_fv])


def FV(state, rider, num_floors):
  rider_fv = [0.0 for _ in range(num_floors)]
  rider_fv[rider.dropoff] = 1.0
  rider_fv = jnp.array(rider_fv)

  elev_fvs = [FVElevator(elev, num_floors) for elev in state]

  return jnp.expand_dims(jnp.hstack([rider_fv] + elev_fvs),
                         axis=0)


def NumFeatures(num_floors, num_elevators):
  # Note: this must match FV(...) above.
  return num_floors + num_elevators * (3 + num_floors + num_floors)


class RLAssigner(object):
  def __init__(self, hparams,  agent=None):
    assert(hparams.nn_sizes or agent), (
      'Must either specify the NN sizes or provide an agent when '
      'initialize RLAssigner.')
    if not agent:
      agent = DQNPureJax(hparams)
    self._agent = agent
    self._hparams = hparams

  def Pick(self, state, rider):
    fv = FV(state, rider, self._hparams.num_floors)
    action = self._agent.Action(fv)
    return action[0]
