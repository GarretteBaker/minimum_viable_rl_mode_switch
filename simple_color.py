from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

@struct.dataclass
class EnvState(environment.EnvState):
    color: int
    time: int  # Renamed from 'step' to 'time' to match your implementation

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_colors: int = 2
    optimal_reward: float = 1.0
    suboptimal_reward: float = 0.25
    max_steps_in_episode: int = 2  # Add this parameter

class SimpleColor(environment.Environment):
    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
            self, 
            key: chex.PRNGKey, 
            state: EnvState, 
            action: int, 
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        reward = jnp.where(state.color == action, 
                           params.optimal_reward, 
                           jnp.where(action == params.max_colors, 
                                     params.suboptimal_reward, 
                                     0.0))
        new_state = EnvState(color=state.color, time=state.time + 1)
        done = self.is_terminal(new_state, params)
        info = {"discount": self.discount(new_state, params)}

        return self.get_obs(new_state), new_state, reward, done, info
    
    def reset_env(
            self, 
            key: chex.PRNGKey, 
            params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        new_color = jax.random.randint(key, shape=(), minval=0, maxval=params.max_colors)
        state = EnvState(color=new_color, time=0)
        return self.get_obs(state), state
    
    def get_obs(self, state: EnvState) -> chex.Array:
        return jnp.array([state.color, state.time], dtype=jnp.float32)
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        return jnp.array(state.time >= params.max_steps_in_episode - 1)
    
    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(params.max_colors + 1)
    
    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(low=0, 
                          high=jnp.array([params.max_colors - 1, params.max_steps_in_episode - 1]), 
                          shape=(2,), 
                          dtype=jnp.float32)