import jax
import jax.numpy as jnp
from simple_color import SimpleColor, EnvParams, EnvState

def create_reward_table(params):
    env = SimpleColor()
    key = jax.random.PRNGKey(0)

    # Create a table header
    header = f"{'Color':^10}|" + "|".join([f"{i:^10}" for i in range(params.max_colors + 1)])
    separator = "-" * len(header)

    print(header)
    print(separator)

    # For each possible color
    for color in range(params.max_colors):
        state = EnvState(color=color, time=0)
        row = [f"{color:^10}|"]
        
        # For each possible action
        for action in range(params.max_colors + 1):
            _, _, reward, _, _ = env.step_env(key, state, action, params)
            row.append(f"{reward:^10.2f}")
        
        print("|".join(row))

    print(separator)
    print(f"{'Action':^10}|" + "|".join([f"{i:^10}" for i in range(params.max_colors + 1)]))

# Set up parameters
params = EnvParams(max_colors=3, optimal_reward=1.0, suboptimal_reward=0.25, max_steps_in_episode=1)

# Create and print the table
create_reward_table(params)