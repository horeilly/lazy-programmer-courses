"""
GridWorld Environment - Pure environment class for reinforcement learning.

This class provides a clean, reusable GridWorld environment following the OpenAI Gym
interface pattern. It separates environment logic (states, actions, transitions, rewards)
from algorithm logic (policies, value functions, learning).

Usage:
    from reinforcement_learning_01.gridworld import GridWorld

    env = GridWorld(config='standard')  # or 'windy'
    state = env.reset()
    next_state, reward, done = env.step('R')

    # Custom configuration
    env = GridWorld.from_config(
        states={(0,0), (0,1), ...},
        actions={(0,0): {'R': {}, 'D': {}}, ...},
        transitions={((0,0), 'R'): {(0,1): 1.0}, ...},
        rewards={(0,3): 1, (1,3): -1, ...},
        terminal_states={(0,3), (1,3)},
        start_state=(2,0),
        discount=0.9
    )
"""

import random
from typing import Any, Dict, Optional, Set, Tuple


class GridWorld:
    """
    Pure environment class for GridWorld - contains ONLY environment logic.

    Attributes:
        states (dict): All valid states {state: None}
        actions (dict): Valid actions per state {state: {action: None}}
        transitions (dict): Transition probabilities {(state, action): {next_state: probability}}
        rewards (dict): Rewards for states {state: reward}
        terminal_states (set): Set of terminal states
        start_state (tuple): Default starting position
        discount (float): Discount factor (gamma)
    """

    def __init__(self, config: str = "standard", **kwargs):
        """
        Initialize GridWorld environment.

        Args:
            config: 'standard' for deterministic grid, 'windy' for stochastic
            **kwargs: Override default parameters (discount, start_state, etc.)
        """
        self._rows = 3
        self._cols = 4
        self._blocked_states = {(1, 1)}
        self._current_state = None

        if config == "standard":
            self._init_standard_grid()
        elif config == "windy":
            self._init_windy_grid()
        else:
            raise ValueError(f"Unknown config: {config}. Use 'standard' or 'windy'")

        # Allow kwargs to override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _init_standard_grid(self):
        """Initialize standard deterministic GridWorld configuration."""
        # States (dict for consistent API)
        self.states = {
            (0, 0): None,
            (0, 1): None,
            (0, 2): None,
            (0, 3): None,
            (1, 0): None,
            (1, 2): None,
            (1, 3): None,
            (2, 0): None,
            (2, 1): None,
            (2, 2): None,
            (2, 3): None,
        }

        # Actions (dict of dicts for consistent API)
        self.actions = {
            (0, 0): {"R": None, "D": None},
            (0, 1): {"L": None, "R": None},
            (0, 2): {"L": None, "R": None, "D": None},
            (0, 3): {},
            (1, 0): {"U": None, "D": None},
            (1, 2): {"U": None, "R": None, "D": None},
            (1, 3): {},
            (2, 0): {"U": None, "R": None},
            (2, 1): {"L": None, "R": None},
            (2, 2): {"U": None, "L": None, "R": None},
            (2, 3): {"U": None, "L": None},
        }

        # Transitions (deterministic)
        self.transitions = {
            # Row 0
            ((0, 0), "R"): {(0, 1): 1.0},
            ((0, 0), "D"): {(1, 0): 1.0},
            ((0, 1), "L"): {(0, 0): 1.0},
            ((0, 1), "R"): {(0, 2): 1.0},
            ((0, 2), "L"): {(0, 1): 1.0},
            ((0, 2), "R"): {(0, 3): 1.0},
            ((0, 2), "D"): {(1, 2): 1.0},
            # Row 1
            ((1, 0), "U"): {(0, 0): 1.0},
            ((1, 0), "D"): {(2, 0): 1.0},
            ((1, 2), "U"): {(0, 2): 1.0},
            ((1, 2), "R"): {(1, 3): 1.0},
            ((1, 2), "D"): {(2, 2): 1.0},
            # Row 2
            ((2, 0), "U"): {(1, 0): 1.0},
            ((2, 0), "R"): {(2, 1): 1.0},
            ((2, 1), "L"): {(2, 0): 1.0},
            ((2, 1), "R"): {(2, 2): 1.0},
            ((2, 2), "U"): {(1, 2): 1.0},
            ((2, 2), "L"): {(2, 1): 1.0},
            ((2, 2), "R"): {(2, 3): 1.0},
            ((2, 3), "U"): {(1, 3): 1.0},
            ((2, 3), "L"): {(2, 2): 1.0},
        }

        # Rewards
        self.rewards = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 1,
            (1, 0): 0,
            (1, 2): 0,
            (1, 3): -1,
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
        }

        # Terminal states
        self.terminal_states = {(0, 3), (1, 3)}

        # Environment parameters
        self.start_state = (2, 0)
        self.discount = 0.9

    def _init_windy_grid(self):
        """Initialize windy (stochastic) GridWorld configuration."""
        # States
        self.states = {
            (0, 0): None,
            (0, 1): None,
            (0, 2): None,
            (0, 3): None,
            (1, 0): None,
            (1, 2): None,
            (1, 3): None,
            (2, 0): None,
            (2, 1): None,
            (2, 2): None,
            (2, 3): None,
        }

        # Actions (same as standard)
        self.actions = {
            (0, 0): {"R": None, "D": None},
            (0, 1): {"L": None, "R": None},
            (0, 2): {"L": None, "R": None, "D": None},
            (0, 3): {},
            (1, 0): {"U": None, "D": None},
            (1, 2): {"U": None, "R": None, "D": None},
            (1, 3): {},
            (2, 0): {"U": None, "R": None},
            (2, 1): {"L": None, "R": None},
            (2, 2): {"U": None, "L": None, "R": None},
            (2, 3): {"U": None, "L": None},
        }

        # Stochastic transitions (extracted from 07_windy_grid_world.py)
        self.transitions = {
            # Row 0 - windy at (0,0)
            ((0, 0), "R"): {(0, 1): 0.7, (0, 2): 0.3},  # Wind blows right
            ((0, 0), "D"): {(1, 0): 1.0},
            ((0, 1), "L"): {(0, 0): 1.0},
            ((0, 1), "R"): {(0, 2): 1.0},
            ((0, 2), "L"): {(0, 1): 1.0},
            ((0, 2), "R"): {(0, 3): 1.0},
            ((0, 2), "D"): {(1, 2): 1.0},
            # Row 1 - windy at (1,2)
            ((1, 0), "U"): {(0, 0): 1.0},
            ((1, 0), "D"): {(2, 0): 1.0},
            ((1, 2), "U"): {(0, 2): 0.5, (1, 3): 0.5},  # Wind interference
            ((1, 2), "R"): {(1, 3): 1.0},
            ((1, 2), "D"): {(2, 2): 1.0},
            # Row 2 - windy at (2,2)
            ((2, 0), "U"): {(1, 0): 1.0},
            ((2, 0), "R"): {(2, 1): 1.0},
            ((2, 1), "L"): {(2, 0): 1.0},
            ((2, 1), "R"): {(2, 2): 1.0},
            ((2, 2), "U"): {(1, 2): 0.5, (2, 3): 0.5},  # Wind interference
            ((2, 2), "L"): {(2, 1): 1.0},
            ((2, 2), "R"): {(2, 3): 1.0},
            ((2, 3), "U"): {(1, 3): 1.0},
            ((2, 3), "L"): {(2, 2): 1.0},
        }

        # Rewards (same as standard)
        self.rewards = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 1,
            (1, 0): 0,
            (1, 2): 0,
            (1, 3): -1,
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
        }

        # Terminal states
        self.terminal_states = {(0, 3), (1, 3)}

        # Environment parameters
        self.start_state = (2, 0)
        self.discount = 0.9

    @classmethod
    def from_config(
        cls,
        states: Dict[Tuple[int, int], Any],
        actions: Dict[Tuple[int, int], Dict[str, Any]],
        transitions: Dict[Tuple[Tuple[int, int], str], Dict[Tuple[int, int], float]],
        rewards: Dict[Tuple[int, int], float],
        terminal_states: Set[Tuple[int, int]],
        start_state: Tuple[int, int],
        discount: float = 0.9,
    ) -> "GridWorld":
        """
        Create GridWorld from explicit configuration.

        Args:
            states: Dict of valid states
            actions: Dict mapping states to dicts of valid actions
            transitions: Dict mapping (state, action) to {next_state: probability}
            rewards: Dict mapping states to rewards
            terminal_states: Set of terminal states
            start_state: Starting position
            discount: Discount factor

        Returns:
            GridWorld instance with custom configuration
        """
        env = cls.__new__(cls)
        env.states = states
        env.actions = actions
        env.transitions = transitions
        env.rewards = rewards
        env.terminal_states = terminal_states
        env.start_state = start_state
        env.discount = discount
        env._current_state = None

        # Infer grid dimensions
        if states:
            max_row = max(s[0] for s in states.keys())
            max_col = max(s[1] for s in states.keys())
            env._rows = max_row + 1
            env._cols = max_col + 1
        else:
            env._rows = 0
            env._cols = 0

        # Infer blocked states
        all_positions = {(r, c) for r in range(env._rows) for c in range(env._cols)}
        env._blocked_states = all_positions - set(states.keys())

        return env

    # === Gym-like Interface ===

    def reset(self, state: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Reset environment to initial state.

        Args:
            state: Optional specific state to reset to (default: start_state)

        Returns:
            Current state after reset
        """
        if state is None:
            self._current_state = self.start_state
        else:
            if state not in self.states:
                raise ValueError(f"Invalid state: {state}")
            self._current_state = state
        return self._current_state

    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action and transition to next state.

        Args:
            action: Action to take ('U', 'D', 'L', 'R')

        Returns:
            (next_state, reward, done) tuple
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step()")

        if action not in self.actions[self._current_state]:
            raise ValueError(f"Invalid action {action} for state {self._current_state}")

        # Sample next state from transition distribution
        next_state = self.sample_transition(self._current_state, action)
        reward = self.rewards[next_state]
        done = self.is_terminal(next_state)

        self._current_state = next_state
        return next_state, reward, done

    # === Query Methods ===

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal."""
        return state in self.terminal_states

    def get_all_states(self) -> Dict[Tuple[int, int], Any]:
        """Return all valid states."""
        return self.states

    def get_possible_actions(self, state: Tuple[int, int]) -> Dict[str, Any]:
        """Get valid actions for a state."""
        return self.actions.get(state, {})

    def get_reward(self, state: Tuple[int, int]) -> float:
        """Get reward for a state."""
        return self.rewards.get(state, 0.0)

    def get_transitions(
        self, state: Tuple[int, int], action: str
    ) -> Dict[Tuple[int, int], float]:
        """Get transition probabilities for (state, action)."""
        return self.transitions.get((state, action), {})

    def sample_transition(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """
        Sample next state from transition distribution.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Next state (sampled from distribution if stochastic)
        """
        transition_probs = self.transitions.get((state, action), {})
        if not transition_probs:
            return state  # Stay in place if no transition defined

        if len(transition_probs) == 1:
            # Deterministic
            return list(transition_probs.keys())[0]

        # Stochastic: sample using cumulative probabilities
        states = list(transition_probs.keys())
        probs = list(transition_probs.values())
        return random.choices(states, weights=probs)[0]

    # === Visualization Methods ===

    def print_values(self, values: Dict[Tuple[int, int], float]) -> None:
        """
        Display value function V(s) in grid format.

        Args:
            values: Dict mapping states to values
        """
        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * self._cols

        print(separator)
        for row in range(self._rows):
            row_values = []
            for col in range(self._cols):
                state = (row, col)
                if state in self._blocked_states:
                    row_values.append(f"{'----':^{cell_width}}")
                elif state in self.states:
                    value = values.get(state, 0.0)
                    row_values.append(f"{value:^{cell_width}.3f}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)

    def print_policy(self, policy: Dict[Tuple[int, int], Any]) -> None:
        """
        Display policy in grid format.

        Handles both deterministic and stochastic policies:
        - Deterministic: policy[state] = 'R'
        - Stochastic: policy[state] = [('R', 0.9), ('D', 0.1)] or {'R': 0.9, 'D': 0.1}

        Args:
            policy: Dict mapping states to actions (or action distributions)
        """
        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * self._cols

        print(separator)
        for row in range(self._rows):
            row_values = []
            for col in range(self._cols):
                state = (row, col)
                if state in self._blocked_states:
                    row_values.append(f"{'----':^{cell_width}}")
                elif state in self.terminal_states:
                    row_values.append(f"{'-':^{cell_width}}")
                elif state in self.states:
                    action_spec = policy.get(state, "-")

                    # Handle different policy formats
                    if isinstance(action_spec, str):
                        # Deterministic: 'R'
                        action_str = action_spec
                    elif isinstance(action_spec, (list, tuple)):
                        # Stochastic list: [('R', 0.9), ('D', 0.1)]
                        if action_spec and isinstance(action_spec[0], (list, tuple)):
                            action_str = max(action_spec, key=lambda x: x[1])[0]
                        else:
                            action_str = str(action_spec)
                    elif isinstance(action_spec, dict):
                        # Stochastic dict: {'R': 0.9, 'D': 0.1}
                        action_str = max(action_spec.items(), key=lambda x: x[1])[0]
                    else:
                        action_str = str(action_spec)

                    row_values.append(f"{action_str:^{cell_width}}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)

    def print_q_values(self, q_table: Dict[Tuple[Tuple[int, int], str], float]) -> None:
        """
        Display Q-values Q(s, a) for each state-action pair.

        Args:
            q_table: Dict mapping (state, action) to Q-values
        """
        print("\nQ-Values (Action Values):")
        print("=" * 60)
        for row in range(self._rows):
            for col in range(self._cols):
                state = (row, col)
                if state in self._blocked_states:
                    print(f"{state}: N/A (blocked)")
                    continue
                if state not in self.states:
                    continue

                print(f"\n{state}:")
                if state in self.terminal_states:
                    print("  Terminal state")
                else:
                    for action in self.actions[state]:
                        q_val = q_table.get((state, action), 0.0)
                        print(f"  {action}: {q_val:.4f}")
        print("=" * 60)

    def render(self) -> None:
        """Display current state position in grid."""
        if self._current_state is None:
            print("Environment not initialized (call reset())")
            return

        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * self._cols

        print(separator)
        for row in range(self._rows):
            row_values = []
            for col in range(self._cols):
                state = (row, col)
                if state in self._blocked_states:
                    row_values.append(f"{'----':^{cell_width}}")
                elif state == self._current_state:
                    row_values.append(f"{'[AGENT]':^{cell_width}}")
                elif state in self.terminal_states:
                    row_values.append(f"{'[TERM]':^{cell_width}}")
                elif state in self.states:
                    row_values.append(f"{'':^{cell_width}}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)


# === Test Script ===

if __name__ == "__main__":
    print("=" * 60)
    print("GridWorld Environment Tests")
    print("=" * 60)

    # Test 1: Basic creation
    print("\nTest 1: Basic Creation")
    print("-" * 60)
    env = GridWorld(config="standard")
    print(f"States: {len(env.get_all_states())} states")
    print(f"Start state: {env.start_state}")
    print(f"Terminal states: {env.terminal_states}")
    print(f"Discount factor: {env.discount}")
    print("✓ Environment created successfully")

    # Test 2: Reset
    print("\nTest 2: Reset")
    print("-" * 60)
    state = env.reset()
    assert state == (2, 0), f"Expected (2, 0), got {state}"
    print(f"Reset to: {state}")

    # Reset to custom state
    state = env.reset(state=(0, 0))
    assert state == (0, 0), f"Expected (0, 0), got {state}"
    print(f"Reset to custom state: {state}")
    print("✓ Reset works correctly")

    # Test 3: Deterministic step
    print("\nTest 3: Deterministic Step")
    print("-" * 60)
    env.reset(state=(0, 0))
    next_state, reward, done = env.step("R")
    assert next_state == (0, 1), f"Expected (0, 1), got {next_state}"
    assert reward == 0.0, f"Expected 0.0, got {reward}"
    assert done == False, f"Expected False, got {done}"
    print(f"(0,0) --R--> {next_state}, reward={reward}, done={done}")
    print("✓ Step works correctly")

    # Test 4: Terminal state detection
    print("\nTest 4: Terminal State Detection")
    print("-" * 60)
    assert env.is_terminal((0, 3)) == True, "Expected (0,3) to be terminal"
    assert env.is_terminal((1, 3)) == True, "Expected (1,3) to be terminal"
    assert env.is_terminal((0, 0)) == False, "Expected (0,0) to be non-terminal"
    print("Terminal: (0,3), (1,3)")
    print("Non-terminal: (0,0)")
    print("✓ Terminal detection works correctly")

    # Test 5: Terminal state step
    print("\nTest 5: Reaching Terminal State")
    print("-" * 60)
    env.reset(state=(0, 2))
    next_state, reward, done = env.step("R")
    assert next_state == (0, 3), f"Expected (0, 3), got {next_state}"
    assert reward == 1.0, f"Expected 1.0, got {reward}"
    assert done == True, f"Expected True, got {done}"
    print(f"(0,2) --R--> {next_state}, reward={reward}, done={done}")
    print("✓ Terminal state transition works correctly")

    # Test 6: Stochastic transitions
    print("\nTest 6: Stochastic Transitions (Windy Grid)")
    print("-" * 60)
    env_windy = GridWorld(config="windy")
    outcomes = {}
    num_trials = 1000

    # Test stochastic transition at (0,0) going right
    for _ in range(num_trials):
        env_windy.reset(state=(0, 0))
        next_state, _, _ = env_windy.step("R")
        outcomes[next_state] = outcomes.get(next_state, 0) + 1

    print(f"(0,0) --R--> outcomes after {num_trials} trials:")
    for state, count in sorted(outcomes.items()):
        prob = count / num_trials
        print(f"  {state}: {count}/{num_trials} ({prob:.2%})")

    assert len(outcomes) > 1, "Expected multiple outcomes (stochastic)"
    print("✓ Stochastic transitions work correctly")

    # Test 7: Query methods
    print("\nTest 7: Query Methods")
    print("-" * 60)
    actions = env.get_possible_actions((0, 0))
    print(f"Actions at (0,0): {list(actions.keys())}")
    assert "R" in actions and "D" in actions, "Expected R and D actions"

    transitions = env.get_transitions((0, 0), "R")
    print(f"Transitions from (0,0) with R: {transitions}")
    assert transitions == {(0, 1): 1.0}, "Expected deterministic transition to (0,1)"

    reward = env.get_reward((0, 3))
    print(f"Reward at (0,3): {reward}")
    assert reward == 1.0, "Expected reward of 1.0"
    print("✓ Query methods work correctly")

    # Test 8: Visualization
    print("\nTest 8: Visualization Methods")
    print("-" * 60)

    # Create dummy policy and values
    policy = {}
    V = {}
    for s in env.get_all_states():
        if not env.is_terminal(s):
            actions = list(env.get_possible_actions(s).keys())
            policy[s] = actions[0] if actions else "-"
        V[s] = 0.5

    print("\nPolicy:")
    env.print_policy(policy)

    print("\nValue Function:")
    env.print_values(V)

    print("\nCurrent Position:")
    env.reset()
    env.render()

    print("✓ Visualization methods work correctly")

    # Test 9: Q-values visualization
    print("\nTest 9: Q-Values Visualization")
    print("-" * 60)
    q_table = {}
    for s in env.get_all_states():
        for a in env.get_possible_actions(s):
            q_table[(s, a)] = random.uniform(0, 1)

    env.print_q_values(q_table)
    print("✓ Q-values visualization works correctly")

    # Test 10: Episode simulation
    print("\nTest 10: Episode Simulation")
    print("-" * 60)
    env.reset()
    print(f"Starting at: {env._current_state}")

    step_count = 0
    max_steps = 20
    while step_count < max_steps:
        actions = list(env.get_possible_actions(env._current_state).keys())
        if not actions:
            print(f"Reached terminal state: {env._current_state}")
            break

        action = random.choice(actions)
        next_state, reward, done = env.step(action)
        step_count += 1
        print(f"  Step {step_count}: {action} -> {next_state} (r={reward})")

        if done:
            print(f"Episode finished in {step_count} steps")
            break

    print("✓ Episode simulation works correctly")

    # Final summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nGridWorld environment is ready for use.")
    print("Import with: from reinforcement_learning_01.gridworld import GridWorld")
