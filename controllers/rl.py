from controllers.agents import AgentParams, DQNPureJax
import jax
import jax.numpy as jnp


def FVElevator(e, num_floors):
  running_fv = jax.nn.one_hot(e['running'].value - 1, 3)
  floor_fv = jax.nn.one_hot(e['floor'], num_floors)
  stops_fv = [0.0 for _ in range(num_floors)]
  for s in e['stops']:
    # s[0] is the immediate committed stop for a rider.
    stops_fv[s[0]] = 1.0
  stops_fv = jnp.array(stops_fv)
  return jnp.hstack([running_fv, floor_fv, stops_fv])


def FV(state, rider, num_floors):
  rider_fv = [0.0 for _ in range(num_floors)]
  rider_fv[rider.dropoff] = 1.0
  rider_fv = jnp.array(rider_fv)

  elev_fvs = [FVElevator(elev, num_floors) for elev in state]

  return jnp.expand_dims(jnp.hstack([rider_fv] + elev_fvs),
                         axis=0)


class RLAssigner(object):
  def __init__(self, num_elevators, num_floors,  agent=None):
    num_features = num_floors + num_elevators * (3 + num_floors + num_floors)
    if not agent:
      params = AgentParams(nn_sizes=[num_features, 30, 30, num_elevators],
                           gamma=0.9,
                           lr=0.001)
      agent = DQNPureJax(params)
    self._agent = agent
    self._num_floors = num_floors

  def Pick(self, state, rider):
    fv = FV(state, rider, self._num_floors)
    action = self._agent.Action(fv)
    return action[0]
