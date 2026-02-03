class GridWorld:
    def __init__(self):
        self.discount = 0.9
        self.epsilon = 0.1
        self.is_active = True
        self.terminal_states = ((0, 3), (1, 3))

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

        self.rewards = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 1,
            (1, 0): 0,
            # (1, 1): 0,
            (1, 2): 0,
            (1, 3): -1,
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
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

        self.values = {
            (0, 0): 0,
            (0, 1): 0,
            (0, 2): 0,
            (0, 3): 0,
            (1, 0): 0,
            # (1, 1): 0,
            (1, 2): 0,
            (1, 3): 0,
            (2, 0): 0,
            (2, 1): 0,
            (2, 2): 0,
            (2, 3): 0,
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
            ((1, 2), "U"): {(0, 2): 1.0},
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
            ((2, 2), "U"): {(1, 2): 1.0},
            ((2, 2), "L"): {(2, 1): 1.0},
            ((2, 2), "R"): {(2, 3): 1.0},
            # (2, 3) transitions
            ((2, 3), "U"): {(1, 3): 1.0},
            ((2, 3), "L"): {(2, 2): 1.0},
        }

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
    v_current = sum([gridworld.values[s] for s in gridworld.values])
    while True:
        gridworld.evaluate_policy()
        gridworld.improve_policy()
        v_pi = sum([gridworld.values[s] for s in gridworld.values])
        if v_pi == v_current:
            break
        v_current = v_pi
    return None


if __name__ == "__main__":
    main()
