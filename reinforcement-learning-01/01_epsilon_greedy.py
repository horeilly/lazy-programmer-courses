from loguru import logger
import numpy as np


NUM_TRIALS = 10000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.25, 0.5, 0.75]


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


def run_experiment(num_trials: int = NUM_TRIALS, epsilon: float = EPSILON):
    # Initialization
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    exploit_index = np.random.randint(len(bandits))
    exploit_bandit = bandits[exploit_index]
    explore_bandits = [bandits[i] for i, _ in enumerate(bandits) if i != exploit_index]

    for i in range(num_trials):
        if np.random.random() < epsilon:
            bandit = explore_bandits[np.random.randint(len(explore_bandits))]
        else:
            bandit = exploit_bandit
        bandit.pull()

        best_explore_bandit = max(bandits, key=lambda b: b.p_estimate)
        if best_explore_bandit.p_estimate > exploit_bandit.p_estimate:
            explore_bandits.append(exploit_bandit)
            explore_bandits.remove(best_explore_bandit)
            exploit_bandit = best_explore_bandit

        if (i + 1) % 1000 == 0:
            logger.info(f"Trial {i + 1}/{num_trials}")
            for idx, b in enumerate(bandits):
                logger.info(
                    f"  Bandit {idx} (p={b.p}): pulls={b.pull_count}, estimate={b.p_estimate:.3f}"
                )


def main():
    run_experiment()
    return None


if __name__ == "__main__":
    main()
