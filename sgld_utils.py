# Heavily taken from https://github.com/edmundlth/validating_lambdahat/blob/main/sgld_utils.py

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import numpy as np
import optax 
from typing import NamedTuple
from llc_utils import param_lp_dist, pack_params, unpack_params

def create_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    else:
        indices = np.arange(len(inputs))

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield inputs[excerpt], targets[excerpt]

class SGLDConfig(NamedTuple):
  epsilon: float
  gamma: float
  num_steps: int
  num_chains: int = 1 
  batch_size: int = None

def mala_acceptance_probability(current_point, proposed_point, loss_and_grad_fn, step_size):
    """
    Calculate the acceptance probability for a MALA transition.

    Args:
    current_point: The current point in parameter space.
    proposed_point: The proposed point in parameter space.
    loss_and_grad_fn (function): Function to compute loss and loss gradient at a point.
    step_size (float): Step size parameter for MALA.

    Returns:
    float: Acceptance probability for the proposed transition.
    """
    # Compute the gradient of the loss at the current point
    current_loss, current_grad = loss_and_grad_fn(current_point)
    proposed_loss, proposed_grad = loss_and_grad_fn(proposed_point)

    # Compute the log of the proposal probabilities (using the Gaussian proposal distribution)
    log_q_proposed_to_current = -jnp.sum((current_point - proposed_point - (step_size * 0.5 * -proposed_grad)) ** 2) / (2 * step_size)
    log_q_current_to_proposed = -jnp.sum((proposed_point - current_point - (step_size * 0.5 * -current_grad)) ** 2) / (2 * step_size)

    # Compute the acceptance probability
    acceptance_log_prob = log_q_proposed_to_current - log_q_current_to_proposed + current_loss - proposed_loss
    return jnp.minimum(1.0, jnp.exp(acceptance_log_prob))

def run_sgld(rngkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=None, trace_batch_loss=True, compute_distance=False, verbose=False):
    num_training_data = len(x_train)
    if itemp is None:
        itemp = 1 / jnp.log(num_training_data)
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=num_training_data,
        w_init=param_init,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.jit(jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0))
    
    sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
    loss_trace = []
    distances = []
    accept_probs = []
    opt_state = sgldoptim.init(param_init)
    param = param_init
    t = 0
    while t < sgld_config.num_steps:
        for x_batch, y_batch in create_minibatches(x_train, y_train, batch_size=sgld_config.batch_size):
            old_param = param.copy()
            _, grads = sgld_grad_fn(param, x_batch, y_batch)
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            if compute_distance: 
                distances.append(param_lp_dist(param_init, param, ord=2))
            if trace_batch_loss:
                loss_trace.append(loss_fn(param, x_batch, y_batch))
            else:
                loss_trace.append(loss_fn(param, x_train, y_train))
            if t % 20 == 0:
                old_param_packed, pack_info = pack_params(old_param)
                param_packed, _ = pack_params(param)
                def grad_fn_packed(w):
                    nll, grad = sgld_grad_fn(unpack_params(w, pack_info), x_batch, y_batch)
                    grad_packed, _ = pack_params(grad)
                    return nll, grad_packed
                prob = mala_acceptance_probability(
                    old_param_packed, 
                    param_packed, 
                    grad_fn_packed, 
                    sgld_config.epsilon
                )
                accept_probs.append((t, prob))
            if t % (sgld_config.num_steps//10) == 0 and verbose:
                print(f"Step {t}, loss: {loss_trace[-1]}")
            t += 1
    return loss_trace, distances, accept_probs

def sgld_run(
        x_train, 
        y_train, 
        batch_size, 
        sgld_grad_fn, 
        sgldoptim, 
        loss_fn, 
        num_steps, 
        opt_state,
        param,
        itemp,
        sgld_config
):
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=x_train.shape[0],
        w_init=param,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0)

    def sgld_step(optstateparam, unused):
        opt_state, param = optstateparam
        i = 0
        def inner_loop(carry, unused):
            i, param, opt_state = carry
            x_batch = jax.lax.dynamic_slice( # TODO: possibly modify these to have randomness, can just replace i with random int
                x_train, 
                (i * batch_size, 0), 
                (batch_size, x_train.shape[1])
            )
            y_batch = jax.lax.dynamic_slice(
                y_train, 
                (i * batch_size, 0), 
                (batch_size, y_train.shape[1])
            )
            _, grads = sgld_grad_fn(param, x_batch, y_batch)
            updates, opt_state = sgldoptim.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            loss = loss_fn(param, x_train, y_train)
            return (i+1, param, opt_state), loss
        (i, param, opt_state), losses = jax.lax.scan(
            inner_loop, (i, param, opt_state), jnp.arange(x_train.shape[0]//batch_size)
        )
        return (opt_state, param), losses
    (opt_state, param), losses = jax.lax.scan(
        sgld_step, (opt_state, param), None, length = num_steps//(x_train.shape[0]//batch_size)+1
    )
    return losses.flatten()

def get_sgld_params(rngkey, loss_fn, sgld_config, param_init, x_train, y_train, itemp=None, trace_batch_loss=True, compute_distance=False, verbose=False):
    num_training_data = len(x_train)
    if itemp is None:
        itemp = 1 / jnp.log(num_training_data)
    local_logprob = create_local_logposterior(
        avgnegloglikelihood_fn=loss_fn,
        num_training_data=x_train.shape[0],
        w_init=param_init,
        gamma=sgld_config.gamma,
        itemp=itemp,
    )
    sgld_grad_fn = jax.value_and_grad(lambda w, x, y: -local_logprob(w, x, y), argnums=0)

    sgldoptim = optim_sgld(sgld_config.epsilon, rngkey)
    loss_trace = []
    distances = []
    accept_probs = []
    opt_state = sgldoptim.init(param_init)
    param = param_init
    batch_size = sgld_config.batch_size
    return (
        x_train, 
        y_train, 
        batch_size, 
        sgld_grad_fn, 
        sgldoptim, 
        loss_fn, 
        sgld_config.num_steps, 
        opt_state, 
        param
    )

    # losses = sgld_run(
    #     x_train, 
    #     y_train, 
    #     batch_size, 
    #     sgld_grad_fn, 
    #     sgldoptim, 
    #     loss_fn, 
    #     sgld_config.num_steps, 
    #     opt_state, 
    #     param
    # )
    # return losses

def generate_rngkey_tree(key_or_seed, tree_or_treedef):
    rngseq = hk.PRNGSequence(key_or_seed)
    return jtree.tree_map(lambda _: next(rngseq), tree_or_treedef)

def optim_sgld(epsilon, rngkey_or_seed):
    @jax.jit
    def sgld_delta(g, rngkey):
        eta = jax.random.normal(rngkey, shape=g.shape) * jnp.sqrt(epsilon)
        return -epsilon * g / 2 + eta

    def init_fn(_):
        return rngkey_or_seed

    @jax.jit
    def update_fn(grads, state):
        rngkey, new_rngkey = jax.random.split(state)
        rngkey_tree = generate_rngkey_tree(rngkey, grads)
        updates = jax.tree_map(sgld_delta, grads, rngkey_tree)
        return updates, new_rngkey
    return optax.GradientTransformation(init_fn, update_fn)


def create_local_logposterior(avgnegloglikelihood_fn, num_training_data, w_init, gamma, itemp):
    def helper(x, y):
        return jnp.sum((x - y)**2)

    def _logprior_fn(w):
        sqnorm = jax.tree_util.tree_map(helper, w, w_init)
        return jax.tree_util.tree_reduce(lambda a,b: a + b, sqnorm)

    def logprob(w, x, y):
        loglike = -num_training_data * avgnegloglikelihood_fn(w, x, y)
        logprior = -gamma / 2 * _logprior_fn(w)
        return itemp * loglike + logprior
    return logprob