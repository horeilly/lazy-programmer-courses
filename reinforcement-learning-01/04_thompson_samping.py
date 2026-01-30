import numpy as np
from loguru import logger

NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p: float):
        self.p = p
        self.p_estimate = 0
        self.pull_count = 0
        self.a = 1
        self.b = 1

    def sample(self) -> float:
        return np.random.beta(self.a, self.b)

    def pull(self):
        reward = int(np.random.random() < self.p)
        self._update(reward)
        return None

    def _update(self, reward: int):
        self.pull_count += 1
        self.a += reward
        self.b += 1 - reward
        self.p_estimate = self.a / (self.a + self.b)


def run_experiment(num_trials: int = NUM_TRIALS) -> None:
    # Initialization
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    for i in range(num_trials):
        bandit = max(bandits, key=lambda b: b.sample())
        bandit.pull()

        if (i + 1) % 1000 == 0:
            logger.info(f"Trial {i + 1}/{num_trials}")
            for idx, b in enumerate(bandits):
                logger.info(
                    f"  Bandit {idx} (p={b.p}): pulls={b.pull_count}, estimate={b.p_estimate:.3f}"
                )

    return None


def main():
    run_experiment()
    return None


if __name__ == "__main__":
    main()
