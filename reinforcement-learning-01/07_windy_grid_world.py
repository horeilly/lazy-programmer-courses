class GridWorld:
    def __init__(self):
        self.discount = 0.9
        self.epsilon = 0.001
        self.is_active = True
        self.state_store = {
            (0, 0): {
                "actions": {
                    "right": {
                        "p": 1.0,
                        "s'": {(0, 1): {"p": 0.7, "r": 0}, (0, 2): {"p": 0.3, "r": 0}},
                    },
                    "down": {"p": 0.0, "s'": {(1, 0): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (0, 1): {
                "actions": {
                    "left": {"p": 0.0, "s'": {(0, 0): {"p": 1, "r": 0}}},
                    "right": {"p": 1.0, "s'": {(0, 2): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (0, 2): {
                "actions": {
                    "left": {"p": 0.0, "s'": {(0, 1): {"p": 1, "r": 0}}},
                    "right": {"p": 1.0, "s'": {(0, 3): {"p": 1, "r": 1}}},
                    "down": {"p": 0.0, "s'": {(1, 2): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (0, 3): {
                "actions": {},
                "value": 0,
            },
            (1, 0): {
                "actions": {
                    "up": {"p": 1.0, "s'": {(0, 0): {"p": 1, "r": 0}}},
                    "down": {"p": 0.0, "s'": {(2, 0): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (1, 2): {
                "actions": {
                    "up": {
                        "p": 1.0,
                        "s'": {(0, 2): {"p": 0.5, "r": 0}, (1, 3): {"p": 0.5, "r": -1}},
                    },
                    "right": {"p": 0.0, "s'": {(1, 3): {"p": 1, "r": -1}}},
                    "down": {"p": 0.0, "s'": {(2, 2): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (1, 3): {"actions": {}, "value": 0.0},
            (2, 0): {
                "actions": {
                    "up": {"p": 0.5, "s'": {(1, 0): {"p": 1, "r": 0}}},
                    "right": {"p": 0.5, "s'": {(2, 1): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (2, 1): {
                "actions": {
                    "left": {"p": 0.0, "s'": {(2, 0): {"p": 1, "r": 0}}},
                    "right": {"p": 1.0, "s'": {(2, 2): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (2, 2): {
                "actions": {
                    "up": {"p": 1.0, "s'": {(1, 2): {"p": 1, "r": 0}}},
                    "left": {"p": 0.0, "s'": {(2, 1): {"p": 1, "r": 0}}},
                    "right": {"p": 0.0, "s'": {(2, 3): {"p": 1, "r": 0}}},
                },
                "value": 0,
            },
            (2, 3): {
                "actions": {
                    "left": {"p": 1.0, "s'": {(2, 2): {"p": 1, "r": 0}}},
                    "up": {"p": 0.0, "s'": {(1, 3): {"p": 1, "r": -1}}},
                },
                "value": 0,
            },
        }

    def value_function(self, state: tuple) -> float:
        value = 0
        for a in self.state_store[state]["actions"]:
            for s_ in self.state_store[state]["actions"][a]["s'"]:
                value += (
                    self.state_store[state]["actions"][a]["p"]
                    * self.state_store[state]["actions"][a]["s'"][s_]["p"]
                    * (
                        self.state_store[state]["actions"][a]["s'"][s_]["r"]
                        + self.discount * self.state_store[s_]["value"]
                    )
                )
        return value

    def update_values(self) -> None:
        delta = 0
        for s in self.state_store:
            v_new = self.value_function(s)
            delta = max([delta, abs(v_new - self.state_store[s]["value"])])
            self.state_store[s]["value"] = v_new
        if delta < self.epsilon:
            self.is_active = False
        return None

    def print_grid(self):
        cell_width = 10
        separator = "+" + (("-" * cell_width) + "+") * 4

        print(separator)
        for row in range(3):
            row_values = []
            for col in range(4):
                if (row, col) == (1, 1):
                    row_values.append(f"{'----':^{cell_width}}")
                elif (row, col) in self.state_store:
                    value = self.state_store[(row, col)]["value"]
                    row_values.append(f"{value:^{cell_width}.3f}")
                else:
                    row_values.append(f"{'----':^{cell_width}}")
            print("|" + "|".join(row_values) + "|")
            print(separator)


def main():
    gridworld = GridWorld()
    while True:
        gridworld.update_values()
        gridworld.print_grid()
        print("\n")
        if not gridworld.is_active:
            break
    return None


if __name__ == "__main__":
    main()
