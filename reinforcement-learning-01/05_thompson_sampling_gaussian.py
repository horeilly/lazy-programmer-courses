import numpy as np
from loguru import logger

NUM_TRIALS = 10000
BANDIT_MUS = [-0.5, 0, 0.5]


class Bandit:
    def __init__(self, mu: float):
        self.mu = mu
        self.tau = 1
        self.pull_count = 0
        self.m = 0
        self.l = 1
        self.sum_x = 0

    def sample(self) -> float:
        return np.random.normal(self.m, 1 / self.l)

    def pull(self) -> None:
        reward = np.random.normal(self.mu, 1 / self.tau)
        self._update(reward)
        return None

    def _update(self, reward: float):
        self.pull_count += 1
        self.l += self.tau
        self.sum_x += reward
        self.m = self.tau * self.sum_x / self.l


def run_experiment(num_trials: int = NUM_TRIALS) -> None:
    # Initialization
    bandits = [Bandit(m) for m in BANDIT_MUS]

    for i in range(num_trials):
        bandit = max(bandits, key=lambda b: b.sample())
        bandit.pull()

        if (i + 1) % 1000 == 0:
            logger.info(f"Trial {i + 1}/{num_trials}")
            for idx, b in enumerate(bandits):
                logger.info(
                    f"  Bandit {idx} (p={b.mu}): pulls={b.pull_count}, estimate={b.m:.3f}"
                )

    return None


def main():
    run_experiment()
    return None


if __name__ == "__main__":
    main()
