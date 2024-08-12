import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jax.experimental import host_callback
import matplotlib.pyplot as plt
import einops
from sgld_utils import run_sgld, SGLDConfig

from color_streak import ColorStreak, EnvParams

import orbax.checkpoint as ocp
from flax.training import orbax_utils
from tqdm import tqdm
# Set up Orbax checkpointer (as before)
checkpointer = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = ocp.CheckpointManager('C:/Users/garre/OneDrive/Documents/AI alignment-MirrorOfScrying/minimum_viable_rl_mode_switch/checkpoints', checkpointer, options)

def save_checkpoint(args):
    params, step = args
    ckpt = {'params': params, 'step': int(step)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(int(step), ckpt, save_kwargs={'save_args': save_args})

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

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = ColorStreak()
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    env_params = config["ENV_PARAMS"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        # Check environment shapes
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        assert obsv.shape[0] == config["NUM_ENVS"], f"Expected first dimension of observation to be {config['NUM_ENVS']}, but got {obsv.shape[0]}"
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        initial_epsilon = 1.0
        final_epsilon = 0.01
        epsilon_decay = (initial_epsilon - final_epsilon) / config["NUM_UPDATES"]

        # TRAIN LOOP
        def _update_step(carry, update_step):
            runner_state, epsilon = carry
            
            # COLLECT TRAJECTORIES
            @jax.jit
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                
                # Epsilon-greedy action selection
                rng, _rng = jax.random.split(rng)
                random_action = jax.random.randint(_rng, shape=(config["NUM_ENVS"],), minval=0, maxval=pi.logits.shape[-1])
                policy_action = pi.sample(seed=_rng)
                
                rng, _rng = jax.random.split(rng)
                epsilon_mask = jax.random.uniform(_rng, shape=(config["NUM_ENVS"],)) < epsilon
                action = jnp.where(epsilon_mask, random_action, policy_action)
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
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, (transition, optimal_actions, suboptimal_actions)
            runner_state, (traj_batch, optimal_actions, suboptimal_actions) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            lambdahat = None
            mala = None
            loss_trace = None
            # CALCULATE b-LLC
            sgld_config = SGLDConfig(
                epsilon = 1e-10, 
                gamma = 10, 
                num_steps = 1000, 
                num_chains = 1, 
                batch_size = 64
            )
            itemp = 1e-3
            @jax.jit
            def loss_fn(params, obsv, targets):
                pi, value = network.apply(params, obsv)
                logits = pi.logits

                pi_targets = targets[..., :-1]
                value_targets = targets[..., -1]

                pi_loss = jnp.linalg.norm(logits - pi_targets)**2
                value_loss = jnp.linalg.norm(value - value_targets)**2
                return jnp.sqrt(pi_loss + value_loss)

            obsv = traj_batch.obs
            obsv = einops.rearrange(obsv, "e s d -> (e s) d")
            pi, val = network.apply(train_state.params, obsv)
            logits = pi.logits

            targets = jnp.concatenate(
                [
                    logits, 
                    jnp.expand_dims(
                        val, 
                        axis = -1
                    )
                ], 
                axis=-1
            )
            rng, sgld_rng = jax.random.split(rng)
            loss_trace, _, mala = run_sgld(
                sgld_rng, 
                loss_fn, 
                sgld_config, 
                train_state.params, 
                obsv, 
                targets, 
                itemp = itemp
            )
            lambdahat = float(np.mean(loss_trace)) * traj_batch.obs.shape[0] * itemp
            mala = np.mean([e[1] for e in mala])

            @jax.jit
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            @jax.jit
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            metric = {
                **traj_batch.info,
                "optimal_action_rate": jnp.sum(optimal_actions) / (config["NUM_STEPS"] * config["NUM_ENVS"]),
                "suboptimal_action_rate": jnp.sum(suboptimal_actions) / (config["NUM_STEPS"] * config["NUM_ENVS"]), 
                "llc": lambdahat, 
                "mala": mala, 
                "loss trace": loss_trace
            }            

            # Checkpointing logic using host_callback
            def checkpoint_if_needed(args):
                params, step = args
                host_callback.id_tap(
                    lambda x, _: save_checkpoint(x),
                    (params, step),
                    result=()
                )

            jax.lax.cond(
                update_step == config["CHECKPOINT_STEP"],
                checkpoint_if_needed,
                lambda _: None,
                (train_state.params, update_step)
            )

            runner_state = (train_state, env_state, last_obs, rng)
            new_epsilon = jnp.maximum(epsilon - epsilon_decay, final_epsilon)
            new_carry = (runner_state, new_epsilon)
            return new_carry, metric

        rng, _rng = jax.random.split(rng)
        initial_carry = ((train_state, env_state, obsv, _rng), initial_epsilon)
        # (runner_state, _), metrics = jax.lax.scan(
        #     _update_step, initial_carry, jnp.arange(config["NUM_UPDATES"], dtype=jnp.int32)
        # )
        metrics = []

        # Perform the equivalent of jax.lax.scan
        runner_state = initial_carry
        update_range = range(int(config["NUM_UPDATES"]))
        for _ in tqdm(update_range):
            runner_state, metric = _update_step(runner_state, None)
            metrics.append(metric)

        # Convert metrics to a numpy array (assuming metrics are numeric)
        metrics = np.stack(metrics)
        runner_state = runner_state[0]

        # Save final checkpoint
        final_step = config["NUM_UPDATES"] - 1
        host_callback.id_tap(
            lambda x, _: save_checkpoint(x),
            (runner_state[0].params, final_step),
            result=()
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

# Set up the configuration
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
    "CHECKPOINT_STEP": 4000,  # New configuration for the checkpoint step
}

# Create the training function
train_fn = make_train(config)

# Run the training
rng = jax.random.PRNGKey(0)
out = train_fn(rng)

# Print the shape and content of the metrics
returns = jnp.mean(out["metrics"]["returned_episode_returns"], axis=1)
optimal_rates = out["metrics"]["optimal_action_rate"]
suboptimal_rates = out["metrics"]["suboptimal_action_rate"]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(returns)
plt.title("Mean Return")
plt.xlabel("Update Step")
plt.ylabel("Return")

plt.subplot(1, 3, 2)
plt.plot(optimal_rates)
plt.title("Optimal Action Rate")
plt.xlabel("Update Step")
plt.ylabel("Rate")

plt.subplot(1, 3, 3)
plt.plot(suboptimal_rates)
plt.title("Suboptimal Action Rate")
plt.xlabel("Update Step")
plt.ylabel("Rate")

plt.tight_layout()
plt.savefig("simple_color_ppo_learning_curve.png")
plt.close()

print("Training complete. Learning curve saved as 'simple_color_ppo_learning_curve.png'")
