import random


class GridWorld:
    def __init__(self):
        self.rows, self.cols = 3, 4
        self.discount = 0.9
        self.epsilon = 0.1
        self.penalty = -0.1
        self.is_active = True
        self.start_state = (2, 0)
        self.current_state = self.start_state
        self.terminal_states = ((0, 3), (1, 3))
        self.rewards = dict()
        self.values = dict()
        self.q_table = dict()

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

        self.policy = dict()
        #     (0, 0): "R",
        #     (0, 1): "R",
        #     (0, 2): "R",
        #     (0, 3): "-",
        #     (1, 0): "U",
        #     # (1, 1): "-",
        #     (1, 2): "R",
        #     (1, 3): "-",
        #     (2, 0): "U",
        #     (2, 1): "R",
        #     (2, 2): "U",
        #     (2, 3): "L",
        # }

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
        rewards = {}
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

    def _randomize_policy(self) -> None:
        for state in self.states:
            if state not in self.terminal_states:
                self.policy[state] = tuple(
                    (a, 1.0 / len(self.actions[state])) for a in self.actions[state]
                )
        return None

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
        # self.current_state = random.choice(self.states)
        # action = self.policy[self.current_state]
        state_actions = []
        # actions = []
        rewards = [0]
        for step in range(max_steps):
            if self.current_state in self.terminal_states:
                state_actions.append((self.current_state, None))
                return state_actions, rewards

            action = self._resolve_policy(self.policy[self.current_state])
            state_actions.append((self.current_state, action))

            self.move(action)
            rewards.append(self.rewards[self.current_state])

        state_actions.append((self.current_state, None))
        return state_actions, rewards

    def run_prediction(self):
        self._randomize_policy()
        Q = {state_action: 0.0 for state_action in self.q_table}
        # print(self.policy)
        returns = {q: [] for q in self.q_table}
        terminal_count = 0
        for episode_num in range(1000):
            state_actions, rewards = self.play_episode()

            # Debug: check if episode reached terminal (terminal rewards are ±1)
            if len(rewards) > 1 and abs(rewards[-1]) == 1:
                terminal_count += 1

            G = 0
            for t in range(len(state_actions) - 2, -1, -1):
                G = rewards[t + 1] + self.discount * G
                if state_actions[t] not in state_actions[:t]:
                    returns[state_actions[t]].append(G)
                    Q[state_actions[t]] = sum(returns[state_actions[t]]) / len(
                        returns[state_actions[t]]
                    )
                    state = state_actions[t][0]
                    eligible_actions = [(sa[1], Q[sa]) for sa in Q if sa[0] == state]
                    # print(eligible_actions)
                    # Random tie-breaking: find max Q-value, then randomly select among ties
                    max_q = max(a[1] for a in eligible_actions)
                    best_actions = [a[0] for a in eligible_actions if a[1] == max_q]
                    policy_update = [
                        (
                            random.choice(best_actions),
                            1 - self.epsilon + (self.epsilon / len(eligible_actions)),
                        )
                    ]
                    for action in eligible_actions:
                        if action[0] != policy_update[0][0]:
                            policy_update.append(
                                (action[0], (self.epsilon / len(eligible_actions)))
                            )
                    self.policy[state] = tuple(policy_update)

            # print(V)
            # self.print_policy()

        print(f"\nDebug: {terminal_count}/1000 episodes reached terminal states")
        print(f"Non-zero Q-values: {sum(1 for v in Q.values() if v != 0.0)}/{len(Q)}")

        # Debug: Print episode length distribution
        episode_lengths = []
        for _ in range(10):
            sa, r = self.play_episode()
            episode_lengths.append(len(sa) - 1)  # Subtract sentinel
        print(f"Sample episode lengths: {episode_lengths}")
        print(f"Sample last rewards: {[self.play_episode()[1][-1] for _ in range(10)]}")

        self.q_table = Q  # Store Q-values for printing
        self.print_q_values()
        self.print_policy()
        return None

    def print_grid(self) -> None:
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
    # gridworld._randomize_policy()
    # print(gridworld.policy)
    gridworld.run_prediction()
    return None


if __name__ == "__main__":
    main()
