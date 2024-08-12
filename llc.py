#%%
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import einops
from typing import Sequence, Tuple, NamedTuple
from gymnax.environments import environment
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import optax
from flax.linen.initializers import constant, orthogonal
import distrax
from itertools import product
from tqdm import tqdm
import textwrap
import matplotlib.pyplot as plt

# Import your custom environment and utilities
from color_streak import ColorStreak, EnvParams
from sgld_utils import run_sgld, SGLDConfig

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

# Redefine the ActorCritic class
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

def load_model(checkpoint_dir: str, step: int) -> dict:
    """Load a model from a checkpoint."""
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)
    return checkpoint_manager.restore(step)

def loss(params, traj_batch, targets):
    pi, value = network.apply(params, traj_batch)
    logits = pi.logits
    
    pi_targets = targets[..., :-1]
    value_targets = targets[..., -1]

    pi_loss = jnp.mean((logits - pi_targets)**2)
    value_loss = jnp.mean((value - value_targets)**2)
    return pi_loss + value_loss

def collect_trajectory_batch(
        params: dict, 
        env: ColorStreak, 
        env_params: EnvParams, 
        num_envs: int, 
        num_steps: int,
        network: nn.Module
    ) -> jnp.ndarray:
    """Collect a trajectory batch without epsilon-greedy action selection."""
    # COLLECT TRAJECTORIES
    def _env_step(runner_state, unused):
        env_state, last_obs, rng = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(params, last_obs)
        
        # Epsilon-greedy action selection
        rng, _rng = jax.random.split(rng)
        # random_action = jax.random.randint(_rng, shape=(config["NUM_ENVS"],), minval=0, maxval=pi.logits.shape[-1])
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, env_state, action, env_params
        )
        
        # Track additional metrics
        optimal_actions = jnp.sum(action == last_obs[..., 0])  # Assuming color is the first element of obs
        suboptimal_actions = jnp.sum(action == env_params.max_colors)
        
        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, (transition, optimal_actions, suboptimal_actions)

    rng = jax.random.PRNGKey(0)
    reset_rng = jax.random.split(rng, num_envs)
    init_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    
    rng, _rng = jax.random.split(rng)
    _, (traj_batch, _, _) = jax.lax.scan(_env_step, (env_state, init_obs, _rng), None, length=num_steps)
    return traj_batch.obs

#%%
params = load_model("checkpoints", 12206)["params"]
env_params = EnvParams(max_colors=2, optimal_reward=1.0, suboptimal_reward=0.25, max_steps_in_episode=10, required_streak=3)
num_envs = 128
steps = 300
env = ColorStreak()
env = FlattenObservationWrapper(env)
env = LogWrapper(env)
network = ActorCritic(env.action_space(env_params).n, activation="relu")
traj_batch = collect_trajectory_batch(params, env, env_params, num_envs, steps, network)
# %%
loss_fn = jax.jit(
    lambda param, inputs, targets: loss(param, inputs, targets)
)
epsilons = [1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
itemps = [0.1, 0.01, 0.001]
gammas = [10.0, 100.0, 1000.0]
results = list()
pbar = tqdm(total=len(epsilons) * len(itemps) * len(gammas))
for epsilon, itemp, gamma in product(epsilons, itemps, gammas):
    sgld_config = SGLDConfig(
        epsilon = epsilon, 
        gamma = gamma, 
        num_steps = 1000, 
        num_chains=1, 
        batch_size=64
    )
    traj_batch_vect = einops.rearrange(traj_batch, "e s d -> (e s) d")

    pi_target, value_target = network.apply(params, traj_batch_vect)
    pi_target = pi_target.logits

    targets = jnp.concatenate(
        [
            pi_target, 
            jnp.expand_dims(
                value_target, 
                axis = -1
            )
        ], 
        axis=-1
    )
    rng = jax.random.key(0)
    loss_trace, _, mala = run_sgld(
        rng, 
        loss_fn, 
        sgld_config, 
        params, 
        traj_batch_vect, 
        targets, 
        itemp=itemp
    )
    init_loss = loss_fn(params, traj_batch_vect, targets)
    lambdahat = float(np.mean(loss_trace) - init_loss) * traj_batch_vect.shape[0] * itemp
    results.append(
        {
            "EPSILON": epsilon, 
            "GAMMA": gamma, 
            "ITEMP": itemp, 
            "LOSS_TRACE": loss_trace, 
            "LLC": lambdahat, 
            "MALA": np.mean([e[1] for e in mala])
        }
    )
    pbar.update(1)
pbar.close()

def create_title(itemp, eps, gamma, num_step, batch_size, lambdahat, mala):
    def format_number(num):
        if abs(num) >= 1000 or (abs(num) < 0.01 and num != 0):
            return f"{num:.2e}"
        else:
            return f"{num:.2f}"

    formatted_params = [
        f"itemp = {format_number(itemp)}",
        f"eps = {format_number(eps)}",
        f"gamma = {format_number(gamma)}",
        f"num_step = {num_step}",
        f"batch_size = {batch_size}",
        f"llc = {format_number(lambdahat)}",
        f"mala = {mala}"
    ]

    return ", ".join(formatted_params)

num_plots = len(results)
num_cols = 5
num_rows = int(np.ceil(num_plots / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
axes = axes.flatten()

pbar = tqdm(total = len(results), desc = f"Creating plot for modelno {12206}")
for i, result in enumerate(results):
    axes[i].plot(
        result["LOSS_TRACE"]
    )
    title = create_title(
        result["ITEMP"], 
        result["EPSILON"], 
        result["GAMMA"], 
        sgld_config.num_steps, 
        sgld_config.batch_size, 
        result["LLC"], 
        result["MALA"]
    )
    max_tit_length = 40
    wrapped_title = textwrap.fill(title, max_tit_length)
    axes[i].set_title(wrapped_title, fontsize=10)
    pbar.update(1)
pbar.close()
plt.tight_layout()
plt.savefig(f"12206_calib.png")
plt.close()

# #%%
# from sgld_utils import create_local_logposterior, create_minibatches


# local_logprob = create_local_logposterior(
#     avgnegloglikelihood_fn=loss_fn,
#     num_training_data=traj_batch_vect.shape[0],
#     w_init=params,
#     gamma=sgld_config.gamma,
#     itemp=itemp,
# )
# sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))
# for x_batch, y_batch in create_minibatches(traj_batch_vect, targets, 64):
#     value, grad = sgld_grad_fn(params, traj_batch_vect, targets)
#     print(f"Grads: {grad}")
#     break
# #%%
# print(lambdahat)
# # %%
# from sgld_utils import create_local_logposterior, create_minibatches, optim_sgld
# rng, rngkey = jax.random.split(rng)
# local_logprob = create_local_logposterior(
#     avgnegloglikelihood_fn=loss_fn,
#     num_training_data=len(traj_batch_vect),
#     w_init=params,
#     gamma=sgld_config.gamma,
#     itemp=itemp,
# )
# sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))

# sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
# loss_trace = []
# distances = []
# accept_probs = []
# opt_state = sgldoptim.init(params)
# param = params
# t = 0
# while t < sgld_config.num_steps:
#     for x_batch, y_batch in create_minibatches(traj_batch_vect, targets, batch_size=sgld_config.batch_size):
#         old_param = param.copy()
#         print(param)
#         loss, grads = sgld_grad_fn(param, x_batch, y_batch)
