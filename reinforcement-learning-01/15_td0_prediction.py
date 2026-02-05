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

    Examples:
        # Uniform random
        action = select_action(env, state)

        # Deterministic
        action = select_action(env, state, policy='R')

        # Stochastic (list)
        action = select_action(env, state, policy=[('R', 0.7), ('D', 0.3)])

        # Stochastic (dict)
        action = select_action(env, state, policy={'R': 0.7, 'D': 0.3})

        # Epsilon-greedy
        action = select_action(env, state, Q=Q_table, epsilon=0.1)
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


def play_episode(env, V, alpha, policy=None, Q=None, epsilon=None, max_steps=20):
    """
    Play one episode using TD(0) prediction.

    Args:
        env: GridWorld environment
        V: Value function dict {state: value}
        alpha: Learning rate (step size)
        policy: Policy to follow (None for uniform random)
        Q: Q-table for epsilon-greedy (optional)
        epsilon: Exploration rate for epsilon-greedy (optional)
        max_steps: Maximum steps per episode

    Returns:
        (steps_taken, max_delta) tuple for convergence monitoring
    """
    state = env.reset()
    max_delta = 0.0
    steps = 0

    for step in range(max_steps):
        # Check if terminal
        if env.is_terminal(state):
            break

        # Select action
        action = select_action(env, state, policy=policy, Q=Q, epsilon=epsilon)
        if action is None:
            break

        # Take step in environment
        next_state, reward, done = env.step(action)

        # TD(0) update: V(S) ← V(S) + α[R + γV(S') - V(S)]
        # Note: Terminal states should have V=0 (no future value)
        v_next = 0.0 if env.is_terminal(next_state) else V[next_state]
        td_target = reward + env.discount * v_next
        td_error = td_target - V[state]

        # Track maximum change for convergence monitoring
        max_delta = max(max_delta, abs(td_error))

        # Update value
        V[state] = V[state] + alpha * td_error

        # Move to next state
        state = next_state
        steps = step + 1

        if done:
            break

    return steps, max_delta


def run_prediction(num_episodes=10000, alpha=0.1, alpha_decay=0.9999,
                   convergence_threshold=0.001, print_frequency=1000,
                   early_stopping=True):
    """
    Run TD(0) prediction with uniform random policy.

    Args:
        num_episodes: Number of episodes to run
        alpha: Initial learning rate (step size)
        alpha_decay: Decay factor for alpha (multiplied each episode)
        convergence_threshold: Stop if value change < threshold
        print_frequency: Print progress every N episodes
        early_stopping: Stop early if converged
    """
    env = GridWorld(config="standard")

    # Initialize value function (terminal states explicitly set to 0)
    V = {state: 0.0 for state in env.get_all_states()}
    V_prev = V.copy()

    print("Running TD(0) Prediction with Uniform Random Policy")
    print("=" * 60)
    print(f"Initial learning rate (alpha): {alpha}")
    print(f"Alpha decay: {alpha_decay}")
    print(f"Discount (gamma): {env.discount}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Convergence threshold: {convergence_threshold}")
    print("=" * 60)

    # Run episodes
    total_steps = 0
    td_errors = []  # TD errors (high variance)
    value_changes = []  # Actual value function changes (for convergence)
    current_alpha = alpha
    converged_count = 0
    episode = 0

    for episode in range(num_episodes):
        # Save previous values for convergence check
        V_prev = V.copy()

        # Play episode and get max TD error
        steps, max_td_error = play_episode(env, V, current_alpha)
        total_steps += steps
        td_errors.append(max_td_error)

        # Ensure terminal states stay at 0
        for terminal_state in env.terminal_states:
            V[terminal_state] = 0.0

        # Calculate actual value function change (for convergence)
        max_value_change = max(abs(V[s] - V_prev[s]) for s in V)
        value_changes.append(max_value_change)

        # Decay learning rate
        current_alpha *= alpha_decay

        # Check convergence based on value changes (not TD errors)
        if max_value_change < convergence_threshold:
            converged_count += 1
        else:
            converged_count = 0

        # Early stopping if converged for 100 consecutive episodes
        if early_stopping and converged_count >= 100:
            print(f"\n✓ Converged at episode {episode + 1}")
            print(f"  Value change: {max_value_change:.6f} < {convergence_threshold}")
            print(f"  Current alpha: {current_alpha:.6f}")
            break

        # Print progress
        if (episode + 1) % print_frequency == 0:
            avg_steps = total_steps / (episode + 1)
            recent_value_change = sum(value_changes[-100:]) / min(100, len(value_changes))
            recent_td_error = sum(td_errors[-100:]) / min(100, len(td_errors))
            print(f"Episode {episode + 1:5d}/{num_episodes} | "
                  f"Steps: {avg_steps:5.2f} | "
                  f"Alpha: {current_alpha:.6f} | "
                  f"Val Δ: {max_value_change:.6f} | "
                  f"TD err: {max_td_error:.4f}")

    episodes_run = episode + 1

    print("\n" + "=" * 60)
    print("Final Value Function:")
    print("=" * 60)
    env.print_values(V)

    print("\n" + "=" * 60)
    print("Convergence Analysis:")
    print("=" * 60)
    print(f"Total episodes run: {episodes_run}")
    print(f"Average episode length: {total_steps / episodes_run:.2f} steps")
    print(f"Final alpha: {current_alpha:.6f}")
    if value_changes:
        print(f"Final value change: {value_changes[-1]:.6f}")
        print(f"Avg value change (last 100): {sum(value_changes[-100:]) / min(100, len(value_changes)):.6f}")
    if td_errors:
        print(f"Final TD error: {td_errors[-1]:.4f}")
        print(f"Avg TD error (last 100): {sum(td_errors[-100:]) / min(100, len(td_errors)):.4f}")

    print("\n" + "=" * 60)
    print("Note: TD errors remain high due to random policy variance.")
    print("Value changes show actual convergence of the value function.")
    print("This evaluates a UNIFORM RANDOM policy (not optimal).")
    print("=" * 60)

    return V, value_changes, td_errors


if __name__ == "__main__":
    # Run TD(0) prediction
    V, value_changes, td_errors = run_prediction(
        num_episodes=10000,
        alpha=0.1,
        alpha_decay=0.9999,
        convergence_threshold=0.001,
        print_frequency=1000,
        early_stopping=True
    )
