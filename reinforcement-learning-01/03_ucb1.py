import numpy as np
from loguru import logger

NUM_TRIALS = 100000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p: float):
        self.p = p
        self.p_estimate = 0
        self.pull_count = 0

    def pull(self):
        reward = int(np.random.random() < self.p)
        self._update(reward)
        return None

    def _update(self, reward: int):
        self.pull_count += 1
        self.p_estimate = (
            self.p_estimate * (self.pull_count - 1) / self.pull_count
        ) + (reward / self.pull_count)


def run_experiment(num_trials: int = NUM_TRIALS):
    # Initialization
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    for bandit in bandits:
        bandit.pull()

    for i in range(len(bandits), num_trials):
        best_bandit = max(
            bandits,
            key=lambda b: (b.p_estimate + np.sqrt(2 * np.log(i) / b.pull_count)),
        )
        best_bandit.pull()

        if (i + 1) % 10000 == 0:
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
