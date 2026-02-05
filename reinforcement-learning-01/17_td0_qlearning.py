"""
Q-Learning - Off-Policy TD Control

Q-learning is an off-policy TD control algorithm that learns the optimal policy
by always updating Q-values based on the maximum Q-value of the next state.

Update rule: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]

Key differences from SARSA:
- Off-policy: Learns optimal policy while following epsilon-greedy behavior policy
- Uses max Q(S',a) - always assumes greedy action will be taken next
- More aggressive/optimistic than SARSA
- Can learn from suboptimal exploration
"""

import importlib.util
import random

# Import GridWorld from 13_gridworld.py
spec = importlib.util.spec_from_file_location(
    "gridworld",
    "/Users/Harry/GitHub/lazy-programmer-courses/reinforcement-learning-01/13_gridworld.py",
)
gridworld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gridworld)
GridWorld = gridworld.GridWorld


def select_action(env, state, policy=None, Q=None, epsilon=None):
    """
    General action selection function.

    Supports multiple policy types:
    - Uniform random (default: policy=None)
    - Deterministic policy (policy='R')
    - Stochastic policy (policy=[('R', 0.7), ('D', 0.3)] or {'R': 0.7, 'D': 0.3})
    - Epsilon-greedy (provide Q and epsilon)

    Args:
        env: GridWorld environment
        state: Current state
        policy: Policy specification (None for uniform random)
        Q: Q-table dict {(state, action): value} (for epsilon-greedy)
        epsilon: Exploration rate (for epsilon-greedy)

    Returns:
        Selected action
    """
    actions = list(env.get_possible_actions(state).keys())

    if not actions:
        return None  # Terminal state

    # Epsilon-greedy (requires Q and epsilon)
    if Q is not None and epsilon is not None:
        if random.random() < epsilon:
            # Explore: uniform random
            return random.choice(actions)
        else:
            # Exploit: greedy action from Q
            return max(actions, key=lambda a: Q.get((state, a), 0.0))

    # No policy specified: uniform random
    if policy is None:
        return random.choice(actions)

    # Deterministic policy: single action (string)
    if isinstance(policy, str):
        return policy

    # Stochastic policy: list of (action, probability) tuples
    if isinstance(policy, (list, tuple)):
        return _sample_from_distribution(policy)

    # Stochastic policy: dict {action: probability}
    if isinstance(policy, dict):
        actions_list = list(policy.keys())
        probs = list(policy.values())
        return random.choices(actions_list, weights=probs)[0]

    raise ValueError(f"Unknown policy format: {type(policy)}")


def _sample_from_distribution(action_probs):
    """
    Sample action from probability distribution using cumulative method.

    Args:
        action_probs: List of (action, probability) tuples

    Returns:
        Sampled action
    """
    choice = random.random()
    cumulative = 0.0

    for action, prob in action_probs:
        cumulative += prob
        if choice < cumulative:
            return action

    # Fallback (handles floating point rounding)
    return action_probs[-1][0]


def qlearning_episode(env, Q, alpha, epsilon, max_steps=100):
    """
    Play one episode using Q-learning.

    Q-learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]

    Args:
        env: GridWorld environment
        Q: Q-table dict {(state, action): value}
        alpha: Learning rate
        epsilon: Exploration rate for epsilon-greedy behavior policy
        max_steps: Maximum steps per episode

    Returns:
        (steps_taken, total_reward, max_q_change) tuple
    """
    # Initialize S
    state = env.reset()

    total_reward = 0
    max_q_change = 0.0
    steps = 0

    for step in range(max_steps):
        if env.is_terminal(state):
            break

        # Choose A from S using behavior policy (epsilon-greedy)
        action = select_action(env, state, Q=Q, epsilon=epsilon)
        if action is None:
            break

        # Take action A, observe R, S'
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Q-learning update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        q_current = Q.get((state, action), 0.0)

        if env.is_terminal(next_state):
            # Terminal state: no future value
            q_next_max = 0.0
        else:
            # Use max_a Q(S',a) - the optimal action (greedy)
            # This is the key difference from SARSA!
            next_actions = list(env.get_possible_actions(next_state).keys())
            if next_actions:
                q_next_max = max(Q.get((next_state, a), 0.0) for a in next_actions)
            else:
                q_next_max = 0.0

        td_target = reward + env.discount * q_next_max
        td_error = td_target - q_current

        # Update Q-value
        Q[(state, action)] = q_current + alpha * td_error

        # Track maximum Q change for convergence
        max_q_change = max(max_q_change, abs(td_error))

        # S ← S'
        state = next_state
        steps = step + 1

        if done:
            break

    return steps, total_reward, max_q_change


def extract_policy(env, Q):
    """
    Extract greedy policy from Q-table.

    Args:
        env: GridWorld environment
        Q: Q-table dict {(state, action): value}

    Returns:
        Policy dict {state: best_action}
    """
    policy = {}

    for state in env.get_all_states():
        if env.is_terminal(state):
            policy[state] = None
            continue

        actions = list(env.get_possible_actions(state).keys())
        if not actions:
            policy[state] = None
            continue

        # Choose action with highest Q-value (greedy)
        best_action = max(actions, key=lambda a: Q.get((state, a), 0.0))
        policy[state] = best_action

    return policy


def run_qlearning(num_episodes=5000, alpha=0.1, epsilon=0.1, alpha_decay=1.0,
                  epsilon_decay=0.9995, min_epsilon=0.01, convergence_threshold=0.001,
                  print_frequency=500, early_stopping=True):
    """
    Run Q-learning to learn optimal policy.

    Args:
        num_episodes: Number of episodes to run
        alpha: Learning rate
        epsilon: Initial exploration rate
        alpha_decay: Decay factor for alpha
        epsilon_decay: Decay factor for epsilon
        min_epsilon: Minimum epsilon value
        convergence_threshold: Stop if Q change < threshold
        print_frequency: Print progress every N episodes
        early_stopping: Stop early if converged

    Returns:
        (Q, policy, rewards, q_changes) tuple
    """
    env = GridWorld(config="standard")

    # Initialize Q-table
    Q = {}
    for state in env.get_all_states():
        for action in env.get_possible_actions(state):
            Q[(state, action)] = 0.0

    print("=" * 70)
    print("Q-Learning - Off-Policy TD Control")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate (alpha): {alpha}")
    print(f"Initial exploration (epsilon): {epsilon}")
    print(f"Epsilon decay: {epsilon_decay} (min: {min_epsilon})")
    print(f"Alpha decay: {alpha_decay}")
    print(f"Discount (gamma): {env.discount}")
    print(f"Convergence threshold: {convergence_threshold}")
    print("=" * 70)

    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    q_changes = []
    current_alpha = alpha
    current_epsilon = epsilon
    converged_count = 0
    Q_prev = Q.copy()

    for episode in range(num_episodes):
        # Save previous Q for convergence check
        Q_prev = Q.copy()

        # Run Q-learning episode
        steps, reward, max_q_change = qlearning_episode(
            env, Q, current_alpha, current_epsilon
        )

        episode_rewards.append(reward)
        episode_lengths.append(steps)
        q_changes.append(max_q_change)

        # Calculate max Q-value change across all state-actions
        max_q_diff = max(abs(Q.get(sa, 0.0) - Q_prev.get(sa, 0.0)) for sa in Q)

        # Decay epsilon and alpha
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)
        current_alpha *= alpha_decay

        # Check convergence
        if max_q_diff < convergence_threshold:
            converged_count += 1
        else:
            converged_count = 0

        # Early stopping
        if early_stopping and converged_count >= 100:
            print(f"\n✓ Converged at episode {episode + 1}")
            print(f"  Max Q change: {max_q_diff:.6f} < {convergence_threshold}")
            break

        # Print progress
        if (episode + 1) % print_frequency == 0:
            recent_rewards = episode_rewards[-100:]
            recent_lengths = episode_lengths[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(recent_lengths) / len(recent_lengths)

            print(f"Episode {episode + 1:5d} | "
                  f"Reward: {reward:6.2f} | "
                  f"Avg(100): {avg_reward:6.2f} | "
                  f"Steps: {steps:3d} | "
                  f"ε: {current_epsilon:.4f} | "
                  f"α: {current_alpha:.4f}")

    episodes_run = episode + 1

    # Extract final policy
    policy = extract_policy(env, Q)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Episodes run: {episodes_run}")
    print(f"Final epsilon: {current_epsilon:.4f}")
    print(f"Final alpha: {current_alpha:.4f}")

    # Calculate average reward over last 100 episodes
    if len(episode_rewards) >= 100:
        avg_reward_final = sum(episode_rewards[-100:]) / 100
        avg_length_final = sum(episode_lengths[-100:]) / 100
        print(f"Average reward (last 100): {avg_reward_final:.2f}")
        print(f"Average length (last 100): {avg_length_final:.2f}")

    print("\n" + "=" * 70)
    print("Learned Optimal Policy:")
    print("=" * 70)
    env.print_policy(policy)

    print("\n" + "=" * 70)
    print("State Values (derived from Q):")
    print("=" * 70)
    # Derive V from Q (V(s) = max_a Q(s,a))
    V = {}
    for state in env.get_all_states():
        if env.is_terminal(state):
            V[state] = 0.0
        else:
            actions = list(env.get_possible_actions(state).keys())
            if actions:
                V[state] = max(Q.get((state, a), 0.0) for a in actions)
            else:
                V[state] = 0.0
    env.print_values(V)

    print("\n" + "=" * 70)
    print("Q-Values (Action Values):")
    print("=" * 70)
    env.print_q_values(Q)

    return Q, policy, episode_rewards, q_changes


def test_policy(env, policy, num_episodes=10):
    """
    Test learned policy without exploration.

    Args:
        env: GridWorld environment
        policy: Learned policy dict
        num_episodes: Number of test episodes

    Returns:
        Average reward
    """
    print("\n" + "=" * 70)
    print(f"Testing Learned Policy ({num_episodes} episodes):")
    print("=" * 70)

    total_reward = 0
    total_steps = 0

    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        trajectory = [state]

        for step in range(100):
            if env.is_terminal(state):
                break

            action = policy[state]
            if action is None:
                break

            next_state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
            trajectory.append(next_state)

            state = next_state

            if done:
                break

        total_reward += episode_reward
        total_steps += steps

        print(f"  Episode {ep + 1}: Reward = {episode_reward:6.2f}, Steps = {steps}, "
              f"Path: {' → '.join(str(s) for s in trajectory)}")

    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes

    print(f"\nAverage reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print("=" * 70)

    return avg_reward


if __name__ == "__main__":
    # Run Q-learning
    Q, policy, rewards, q_changes = run_qlearning(
        num_episodes=5000,
        alpha=0.1,
        epsilon=0.1,
        alpha_decay=0.9999,
        epsilon_decay=0.9995,
        min_epsilon=0.01,
        convergence_threshold=0.001,
        print_frequency=500,
        early_stopping=True
    )

    # Test the learned policy
    env = GridWorld(config="standard")
    test_policy(env, policy, num_episodes=10)

    print("\n" + "=" * 70)
    print("Q-Learning completed successfully!")
    print("The learned policy should navigate optimally to the +1 reward")
    print("while avoiding the -1 penalty.")
    print("\n" + "=" * 70)
    print("Key Difference from SARSA:")
    print("  SARSA (on-policy):  Uses Q(S',A') where A' is actually taken")
    print("  Q-learning (off-policy): Uses max_a Q(S',a) - always greedy")
    print("=" * 70)
