import numpy as np

def estimate_steps_to_optimal(n_colors, required_streak, optimal_reward, suboptimal_reward, max_steps_per_episode):
    # Expected reward from suboptimal action
    suboptimal_expected = suboptimal_reward

    # Probability of choosing the correct color
    p_correct = 1 / n_colors

    # Expected number of attempts to achieve the required streak
    expected_attempts = (1 / (p_correct ** required_streak))

    # Expected reward from optimal strategy per episode
    optimal_expected = (optimal_reward * (max_steps_per_episode - required_streak)) / max_steps_per_episode

    # Expected number of episodes to clearly distinguish optimal from suboptimal
    episodes_to_distinguish = np.ceil(1 / (optimal_expected - suboptimal_expected))

    # Total steps estimate
    total_steps_estimate = episodes_to_distinguish * max_steps_per_episode * expected_attempts

    return total_steps_estimate

# Example usage
n_colors = 3
required_streak = 3
optimal_reward = 1.0
suboptimal_reward = 0.5
max_steps_per_episode = 10

estimate = estimate_steps_to_optimal(n_colors, required_streak, optimal_reward, suboptimal_reward, max_steps_per_episode)
print(f"Estimated steps to figure out optimal strategy: {estimate}")

# Let's calculate for a range of required_streak values
for streak in range(1, 6):
    estimate = estimate_steps_to_optimal(n_colors, streak, optimal_reward, suboptimal_reward, max_steps_per_episode)
    print(f"For required_streak = {streak}: Estimated steps = {estimate}")