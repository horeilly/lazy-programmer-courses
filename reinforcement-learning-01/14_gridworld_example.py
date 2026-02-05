"""
Example: Using the GridWorld Environment Base Class

This demonstrates how future RL algorithms should use the GridWorld environment.
The environment handles all state transitions, while the algorithm manages policy and values.
"""

import sys
import importlib.util

# Import GridWorld from 13_gridworld.py
spec = importlib.util.spec_from_file_location(
    "gridworld",
    "/Users/Harry/GitHub/lazy-programmer-courses/reinforcement-learning-01/13_gridworld.py"
)
gridworld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gridworld)
GridWorld = gridworld.GridWorld


def simple_policy_evaluation_example():
    """
    Example: Simple policy evaluation using GridWorld environment.

    This shows the clean separation:
    - Environment (GridWorld) manages states, actions, transitions, rewards
    - Algorithm manages policy and value function
    """
    print("=" * 70)
    print("GridWorld Environment Usage Example")
    print("=" * 70)

    # Create environment
    env = GridWorld(config='standard')
    print(f"\n✓ Environment created: {len(env.get_all_states())} states")

    # Algorithm owns policy and values (as dicts for consistency)
    print("\n1. Initialize policy (algorithm logic - NOT in environment)")
    policy = {}
    for s in env.get_all_states():
        if not env.is_terminal(s):
            # Pick first available action as initial policy
            actions = list(env.get_possible_actions(s).keys())
            policy[s] = actions[0] if actions else None

    print("   Initial policy created (random first action)")

    # Initialize value function (algorithm owns this)
    print("\n2. Initialize value function (algorithm logic - NOT in environment)")
    V = {s: 0.0 for s in env.get_all_states()}
    print("   Value function initialized to zeros")

    # Policy evaluation (algorithm uses environment for queries)
    print("\n3. Policy evaluation (algorithm queries environment)")
    print("   Algorithm iterates and updates V(s) using environment data")

    epsilon = 0.1
    max_iterations = 100

    for iteration in range(max_iterations):
        delta = 0
        for s in env.get_all_states():
            if env.is_terminal(s):
                continue

            action = policy[s]
            if action is None:
                continue

            # Query environment for transitions
            v_new = 0
            for s_next, prob in env.get_transitions(s, action).items():
                reward = env.get_reward(s_next)
                v_new += prob * (reward + env.discount * V[s_next])

            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < epsilon:
            print(f"   Converged in {iteration + 1} iterations (delta={delta:.6f})")
            break

    # Visualization using environment utilities
    print("\n4. Display results using environment visualization methods")
    print("\nPolicy:")
    env.print_policy(policy)

    print("\nValue Function:")
    env.print_values(V)

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("✓ Environment manages: states, actions, transitions, rewards")
    print("✓ Algorithm manages: policy, values, learning logic")
    print("✓ Clean separation makes code reusable and maintainable")
    print("=" * 70)


def episode_simulation_example():
    """
    Example: Simulating episodes using the Gym-like interface.
    """
    print("\n\n" + "=" * 70)
    print("Episode Simulation Example (Gym-like Interface)")
    print("=" * 70)

    env = GridWorld(config='standard')

    # Simple random policy for demonstration
    import random

    print("\nRunning episode with random policy...")
    state = env.reset()
    print(f"Starting state: {state}")

    episode_states = [state]
    episode_rewards = []
    max_steps = 20

    for step in range(max_steps):
        # Get valid actions
        actions = list(env.get_possible_actions(state).keys())
        if not actions:
            print(f"\nReached terminal state: {state}")
            break

        # Random policy
        action = random.choice(actions)

        # Step in environment
        next_state, reward, done = env.step(action)

        print(f"  Step {step + 1}: {state} --{action}--> {next_state} (r={reward})")

        episode_states.append(next_state)
        episode_rewards.append(reward)

        if done:
            print(f"\nEpisode finished in {step + 1} steps")
            print(f"Total return: {sum(episode_rewards)}")
            break

        state = next_state

    print("\n" + "=" * 70)


def stochastic_environment_example():
    """
    Example: Working with stochastic (windy) environment.
    """
    print("\n\n" + "=" * 70)
    print("Stochastic Environment Example")
    print("=" * 70)

    env = GridWorld(config='windy')

    print("\nTesting stochastic transition at (0,0) going right:")
    print("Expected: 70% to (0,1), 30% to (0,2)")

    outcomes = {}
    num_trials = 1000

    for _ in range(num_trials):
        env.reset(state=(0, 0))
        next_state, _, _ = env.step('R')
        outcomes[next_state] = outcomes.get(next_state, 0) + 1

    print(f"\nResults after {num_trials} trials:")
    for state, count in sorted(outcomes.items()):
        percentage = (count / num_trials) * 100
        print(f"  {state}: {count:4d}/{num_trials} ({percentage:5.2f}%)")

    print("\n✓ Environment correctly handles stochastic transitions")
    print("=" * 70)


if __name__ == "__main__":
    # Run examples
    simple_policy_evaluation_example()
    episode_simulation_example()
    stochastic_environment_example()

    print("\n\n" + "=" * 70)
    print("Future Usage:")
    print("=" * 70)
    print("""
Future RL algorithms (TD learning, Q-learning, etc.) should:

1. Import GridWorld:
   from gridworld import GridWorld  # When file is renamed
   # OR use importlib as shown above for 13_gridworld.py

2. Create environment:
   env = GridWorld(config='standard')  # or 'windy'

3. Implement algorithm logic:
   - Manage policy, values, Q-table (algorithm owns these)
   - Query environment for transitions: env.get_transitions(s, a)
   - Query environment for rewards: env.get_reward(s)
   - Use Gym interface: env.reset(), env.step(action)

4. Use environment visualization:
   - env.print_policy(policy)
   - env.print_values(V)
   - env.print_q_values(Q)
   - env.render()

This separation keeps code clean, testable, and reusable!
    """)
    print("=" * 70)
