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

# Define the configuration dictionary
config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 64,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e8,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_PARAMS": EnvParams(max_colors=2, optimal_reward=1.0, suboptimal_reward=0.25, max_steps_in_episode=10, required_streak=3),
    "ANNEAL_LR": True,
    "CHECKPOINT_STEP": 4000,
}

def load_model(checkpoint_dir: str, step: int) -> dict:
    """Load a model from a checkpoint."""
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, checkpointer)
    return checkpoint_manager.restore(step)

def collect_trajectory_batch(params: dict, env: ColorStreak, env_params: EnvParams, config: dict, network: nn.Module) -> jnp.ndarray:
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
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, env_state, action, env_params
        )
        
        # Track additional metrics
        optimal_actions = jnp.sum(action == last_obs[..., 0])  # Assuming color is the first element of obs
        suboptimal_actions = jnp.sum(action == config["ENV_PARAMS"].max_colors)
        
        transition = Transition(
            done, action, value, reward, log_prob, last_obs, info
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, (transition, optimal_actions, suboptimal_actions)

    rng = jax.random.PRNGKey(0)
    reset_rng = jax.random.split(rng, config["NUM_ENVS"])
    init_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    
    rng, _rng = jax.random.split(rng)
    _, (traj_batch, _, _) = jax.lax.scan(_env_step, (env_state, init_obs, _rng), None, length=config["NUM_STEPS"])
    return traj_batch.obs

def calculate_llc(params: dict, config: dict, env: ColorStreak, env_params: EnvParams,
                  epsilon: float = 1e-4, gamma: float = 10, batch_size: int = 4,
                  itemp: float = 1e-3, num_steps: int = 1000) -> Tuple[float, float, np.ndarray]:
    """
    Calculate the LLC for a given model with specified SGLD hyperparameters.
    
    Args:
        params: Model parameters
        config: Configuration dictionary
        env: ColorStreak environment
        env_params: Environment parameters
        epsilon: SGLD step size
        gamma: SGLD momentum parameter
        batch_size: SGLD batch size
        itemp: Inverse temperature for SGLD
        num_steps: Number of SGLD steps
    
    Returns:
        lambdahat: Estimated Local Lipschitz Constant
        mala: Mean Absolute Local Approximation
        loss_trace: Array of loss values during SGLD
    """
    network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
    
    sgld_config = SGLDConfig(
        epsilon=epsilon,
        gamma=gamma,
        num_steps=num_steps,
        num_chains=1,
        batch_size=batch_size
    )

    def loss(params, obsv, target):
        pi, value = network.apply(params, obsv)
        logits = pi.probs

        pi_targets = target[..., :-1]
        print(pi_targets.shape)
        value_targets = target[..., -1]
        print(value_targets.shape)

        pi_loss = jnp.linalg.norm(logits - pi_targets)**2
        value_loss = jnp.linalg.norm(value - value_targets)**2
        return pi_loss + value_loss
    loss_fn = jax.jit(lambda param, input, target: loss(param, input, target))

    # Collect trajectory batch
    obsv = collect_trajectory_batch(params, env, env_params, config, network)
    obsv = einops.rearrange(obsv, "t e d -> (e t) d")
    print(f"Observation shape: {obsv.shape}")
    print(obsv)

    # Generate targets
    pi, val = network.apply(params, obsv)
    logits = pi.probs
    targets = jnp.concatenate([logits, jnp.expand_dims(val, axis=-1)], axis=-1)
    print(f"Targets shape: {targets.shape}")

    rng = jax.random.PRNGKey(0)
    rng, sgld_rng = jax.random.split(rng)
    loss_trace, _, mala = run_sgld(
        sgld_rng,
        loss_fn,
        sgld_config,
        params,
        obsv,
        targets,
        itemp=itemp
    )
    initial_loss = loss_fn(params, obsv, targets)
    print(f"initial loss: {initial_loss}")
    lambdahat = float(np.mean(loss_trace)) * obsv.shape[0] * itemp
    mala = np.mean([e[1] for e in mala])

    return lambdahat, mala, np.array(loss_trace)

def hyperparams_test(checkpoint_dir: str, step: int, config: dict):
    env = ColorStreak()
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = config["ENV_PARAMS"]

    model = load_model(checkpoint_dir, step)

    epsilons = [1e-2]
    gammas = [10]
    batches = [64]
    itemps = [0.01]
    results = list()

    combo_len = len(epsilons) * len(gammas) * len(batches) * len(itemps)
    pbar = tqdm(total=combo_len, desc=f"Parameter Search Progress on {step}")
    for epsilon, gamma, batch, itemp in product(epsilons, gammas, batches, itemps):
        llc, mala, loss_trace = calculate_llc(
            model["params"], 
            config, 
            env, 
            env_params, 
            epsilon = epsilon, 
            gamma = gamma, 
            batch_size = batch, 
            itemp = itemp, 
            num_steps=1e3
        )
        results.append(
            {
                "epsilon": epsilon, 
                "gamma": gamma, 
                "batch_size": batch, 
                "itemp": itemp, 
                "llc": llc, 
                "mala": mala, 
                "loss_trace": loss_trace
            }
        )
        pbar.update(1)
    pbar.close()

    num_plots = len(results)
    num_cols = 5
    num_rows = int(np.ceil(num_plots / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    axes = axes.flatten()

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

    pbar = tqdm(total = len(results), desc = f"Creating plot for modelno {step}")
    for i, result in enumerate(results):
        axes[i].plot(
            result["loss_trace"]
        )
        title = create_title(
            result["itemp"], 
            result["epsilon"], 
            result["gamma"], 
            1000, 
            result["batch_size"], 
            result["llc"],
            result["mala"]
        )
        max_title_length = 40
        wrapped_title = textwrap.fill(title, max_title_length)
        axes[i].set_title(wrapped_title, fontsize=10)
        pbar.update(1)
    pbar.close()
    print(f"Saving plot")
    folder = "C:/Users/garre/OneDrive/Documents/AI alignment-MirrorOfScrying/minimum_viable_rl_mode_switch"
    plt.tight_layout()
    plt.savefig(f"{folder}/{step}_behavioral_llc_calib.png")
    plt.close()



def compare_models_llc(checkpoint_dir: str, step1: int, step2: int, config: dict):
    """Load two models and compare their LLCs."""
    env = ColorStreak()
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = config["ENV_PARAMS"]

    model1 = load_model(checkpoint_dir, step1)
    model2 = load_model(checkpoint_dir, step2)

    llc1, mala1 = calculate_llc(model1['params'], config, env, env_params)
    llc2, mala2 = calculate_llc(model2['params'], config, env, env_params)

    print(f"Model at step {step1}:")
    print(f"  LLC: {llc1}")
    print(f"  MALA: {mala1}")
    print(f"Model at step {step2}:")
    print(f"  LLC: {llc2}")
    print(f"  MALA: {mala2}")

if __name__ == "__main__":
    checkpoint_dir = 'C:/Users/garre/OneDrive/Documents/AI alignment-MirrorOfScrying/minimum_viable_rl_mode_switch/checkpoints'
    step1 = 4000  # Replace with your actual checkpoint step
    step2 = int(config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])) - 1  # Final step

    # compare_models_llc(checkpoint_dir, step1, step2, config)
    hyperparams_test(checkpoint_dir, step1, config)
    hyperparams_test(checkpoint_dir, step2, config)