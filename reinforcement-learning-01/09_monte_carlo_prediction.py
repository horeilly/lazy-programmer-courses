import random


class GridWorld:
    def __init__(self):
        self.rows, self.cols = 3, 4
        self.discount = 0.9
        self.epsilon = 0.1
        self.penalty = 0.0
        self.is_active = True
        self.start_state = (2, 0)
        self.current_state = self.start_state
        self.terminal_states = ((0, 3), (1, 3))
        self.rewards = dict()
        self.values = dict()

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

        self.policy = {
            (0, 0): "R",
            (0, 1): "R",
            (0, 2): "R",
            (0, 3): "-",
            (1, 0): "U",
            # (1, 1): "-",
            (1, 2): "R",
            (1, 3): "-",
            (2, 0): "U",
            (2, 1): "R",
            (2, 2): "U",
            (2, 3): "L",
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

    def _initialize_values(self) -> None:
        self.values = {state: 0 for state in self.states}
        return None

    def _set_state(self, state: tuple) -> None:
        self.current_state = state
        return None

    @staticmethod
    def _resolve_transition(p: dict[tuple[int, int], float]) -> tuple[int, int]:
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

    def play_episode(self, max_steps: int = 20) -> tuple[list, list, list]:
        # self._set_state(self.start_state)
        self.current_state = random.choice(self.states)
        states = []
        actions = []
        rewards = [0]
        for step in range(max_steps):
            states.append(self.current_state)
            if self.current_state in self.terminal_states:
                return states, actions, rewards

            action = self.policy[self.current_state]
            actions.append(action)

            self.move(action)
            rewards.append(self.rewards[self.current_state])
        return states, actions, rewards

    def run_prediction(self):
        V = {state: 0.0 for state in self.states}
        returns = {state: [] for state in self.states}
        for _ in range(100):
            states, actions, rewards = self.play_episode()
            G = 0
            for t in range(len(states) - 2, -1, -1):
                G = rewards[t + 1] + self.discount * G
                if states[t] not in states[:t]:
                    returns[states[t]].append(G)
                    V[states[t]] = sum(returns[states[t]]) / len(returns[states[t]])
            # print(V)
            # print(returns)
        self.values = V
        self.print_grid()
        self.print_policy()
        return None

    def value_function(self, state: tuple, action: str) -> float:
        value = 0
        for s_ in self.transitions[(state, action)]:
            p_s_r_s_a = self.transitions[(state, action)][s_]
            r = self.rewards[s_]
            v_s = self.values[s_]

            value += p_s_r_s_a * (r + self.discount * v_s)

        return value

    def update_values(self) -> None:
        delta = 0
        for s in self.states:
            if s in self.terminal_states:
                continue
            v_new = self.value_function(s, self.policy[s])
            delta = max([delta, abs(v_new - self.values[s])])
            self.values[s] = v_new
        if delta < self.epsilon:
            self.is_active = False
        return None

    def evaluate_policy(self) -> None:
        self.print_policy()
        print("\n")
        while True:
            self.update_values()
            self.print_grid()
            print("\n")
            if not self.is_active:
                break
        return None

    def improve_policy(self) -> None:
        is_policy_stable = True
        while True:
            for s in self.states:
                if s in self.terminal_states:
                    continue
                pi_current = self.policy[s]
                pi = pi_current
                v_s_a_current = self.value_function(s, pi)
                for a in self.actions[s]:
                    v_s_a_proposed = self.value_function(s, a)
                    if v_s_a_proposed > v_s_a_current:
                        v_s_a_current = v_s_a_proposed
                        pi = a
                    self.policy[s] = pi
                self.print_policy()
                print("\n")
                if pi_current != pi:
                    is_policy_stable = False
                else:
                    is_policy_stable = True
            if is_policy_stable:
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
                    action = self.policy[(row, col)]
                    row_values.append(f"{action:^{cell_width}}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)
        return None


def main():
    gridworld = GridWorld()
    gridworld.run_prediction()
    return None


if __name__ == "__main__":
    main()
