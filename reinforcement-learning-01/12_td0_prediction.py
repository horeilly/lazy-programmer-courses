import random


class GridWorld:
    def __init__(self):
        self.rows, self.cols = 3, 4
        self.discount = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.penalty = -0.1
        self.is_active = True
        self.start_state = (2, 0)
        self.current_state = self.start_state
        self.terminal_states = ((0, 3), (1, 3))
        self.rewards = dict()
        self.values = dict()
        self.q_table = dict()
        self.policy = dict()

        self.states = (
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            # (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
        )

        self.actions = {
            (0, 0): ("R", "D"),
            (0, 1): ("L", "R"),
            (0, 2): ("L", "R", "D"),
            (0, 3): (),
            (1, 0): ("U", "D"),
            # (1, 1): (),
            (1, 2): ("U", "R", "D"),
            (1, 3): (),
            (2, 0): ("U", "R"),
            (2, 1): ("L", "R"),
            (2, 2): ("U", "L", "R"),
            (2, 3): ("U", "L"),
        }

        self.transitions = {
            # (0, 0) transitions
            ((0, 0), "R"): {(0, 1): 1.0},
            ((0, 0), "D"): {(1, 0): 1.0},
            # (0, 1) transitions
            ((0, 1), "L"): {(0, 0): 1.0},
            ((0, 1), "R"): {(0, 2): 1.0},
            # (0, 2) transitions
            ((0, 2), "L"): {(0, 1): 1.0},
            ((0, 2), "R"): {(0, 3): 1.0},
            ((0, 2), "D"): {(1, 2): 1.0},
            # (0, 3) is terminal, no transitions
            # (1, 0) transitions
            ((1, 0), "U"): {(0, 0): 1.0},
            ((1, 0), "D"): {(2, 0): 1.0},
            # (1, 1) doesn't exist
            # (1, 2) transitions
            ((1, 2), "U"): {(0, 2): 0.5, (1, 3): 0.5},
            ((1, 2), "R"): {(1, 3): 1.0},
            ((1, 2), "D"): {(2, 2): 1.0},
            # (1, 3) is terminal, no transitions
            # (2, 0) transitions
            ((2, 0), "U"): {(1, 0): 1.0},
            ((2, 0), "R"): {(2, 1): 1.0},
            # (2, 1) transitions
            ((2, 1), "L"): {(2, 0): 1.0},
            ((2, 1), "R"): {(2, 2): 1.0},
            # (2, 2) transitions
            ((2, 2), "U"): {(1, 2): 0.5, (2, 3): 0.5},
            ((2, 2), "L"): {(2, 1): 1.0},
            ((2, 2), "R"): {(2, 3): 1.0},
            # (2, 3) transitions
            ((2, 3), "U"): {(1, 3): 1.0},
            ((2, 3), "L"): {(2, 2): 1.0},
        }

        self._build_rewards(self.penalty)
        self._initialize_values()
        self._build_q_table()

    def _build_states(self) -> None:
        self.states = tuple(
            [(i, j) for i in range(3) for j in range(4) if (i, j) != (1, 1)]
        )

    def _build_rewards(self, penalty: float = 0) -> None:
        rewards = dict()
        for state in self.states:
            if state == (0, 3):
                rewards[state] = 1
            elif state == (1, 3):
                rewards[state] = -1
            else:
                rewards[state] = penalty
        self.rewards = rewards
        return None

    def _build_q_table(self) -> None:
        self.q_table = {}
        for state in self.actions:
            for action in self.actions[state]:
                self.q_table[(state, action)] = 0.0
        return None

    def _set_policy(self, policy: dict) -> None:
        self.policy = policy
        return None

    def _randomize_policy(self) -> dict:
        policy = dict()
        for state in self.states:
            if state not in self.terminal_states:
                policy[state] = tuple(
                    (a, 1.0 / len(self.actions[state])) for a in self.actions[state]
                )
        return policy

    def _initialize_values(self) -> None:
        self.values = {state: 0 for state in self.states}
        return None

    def _set_state(self, state: tuple) -> None:
        self.current_state = state
        return None

    @staticmethod
    def _resolve_policy(p: dict[tuple[int, int], tuple]) -> str:
        choice = random.random()
        threshold = 0
        for action, probability in p:
            threshold += probability
            if choice < threshold:
                return action
        raise ValueError("Implementation error")

    @staticmethod
    def _resolve_transition(p: dict[tuple[int, int], tuple]) -> tuple[int, int]:
        choice = random.random()
        threshold = 0
        for s in p:
            threshold += p[s]
            if choice < threshold:
                return s
        raise ValueError("Implementation error")

    def move(self, action: str):
        next_state_probs = self.transitions[(self.current_state, action)]
        next_state = self._resolve_transition(next_state_probs)
        self._set_state(next_state)
        return None

    def play_episode(self, max_steps: int = 20) -> tuple[list, list]:
        self._set_state(self.start_state)
        for step in range(max_steps):
            if self.current_state in self.terminal_states:
                break
            s = self.current_state
            action = self._resolve_policy(self.policy[self.current_state])
            self.move(action)
            s2, r = self.current_state, self.rewards[self.current_state]
            self.values[s] += self.alpha * (
                r + self.discount * self.values[s2] - self.values[s]
            )

    def run_prediction(self):
        policy = self._randomize_policy()
        self._set_policy(policy)
        self._initialize_values()
        for episode in range(10000):
            self.play_episode()
            if (episode + 1) % 1000 == 0:
                print(f"\nAfter episode {episode + 1}:")
                self.print_values()
        return None

    def print_values(self) -> None:
        """Print state values in a grid format."""
        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * 4

        print(separator)
        for row in range(3):
            row_values = []
            for col in range(4):
                if (row, col) == (1, 1):
                    row_values.append(f"{'----':^{cell_width}}")
                elif (row, col) in self.states:
                    value = self.values[(row, col)]
                    row_values.append(f"{value:^{cell_width}.3f}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)
        return None

    def print_grid(self) -> None:
        """Alias for print_values."""
        self.print_values()
        return None

    def print_policy(self) -> None:
        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * 4

        print(separator)
        for row in range(3):
            row_values = []
            for col in range(4):
                if (row, col) == (1, 1):
                    row_values.append(f"{'----':^{cell_width}}")
                elif (row, col) in self.states:
                    if (row, col) in self.terminal_states:
                        row_values.append(f"{'-':^{cell_width}}")
                    else:
                        # Get action with highest probability (argmax)
                        policy_probs = self.policy[(row, col)]
                        best_action = max(policy_probs, key=lambda x: x[1])[0]
                        row_values.append(f"{best_action:^{cell_width}}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)
        return None

    def print_q_values(self) -> None:
        """Print Q-values (action values) for each state."""
        print("\nQ-Values (Action Values):")
        print("=" * 60)
        for row in range(3):
            for col in range(4):
                state = (row, col)
                if state == (1, 1):
                    print(f"{state}: N/A (blocked)")
                    continue
                if state not in self.states:
                    continue

                print(f"\n{state}:")
                if state in self.terminal_states:
                    print("  Terminal state")
                else:
                    for action in self.actions[state]:
                        q_val = self.q_table.get((state, action), 0.0)
                        print(f"  {action}: {q_val:.4f}")
        print("=" * 60)
        return None


def main():
    gridworld = GridWorld()
    gridworld.run_prediction()
    return None


if __name__ == "__main__":
    main()
