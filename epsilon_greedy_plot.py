from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from oosapy.models import List

from bandit import Bandit
from bandit_utilss import logger
from epsilon_greedy_agent import EpsilonGreedyAgent


def compare_epsilons(
        epsilons: List[float],
        bandits_true_means: List[float],
        iterations: int,
) -> Tuple[List[EpsilonGreedyAgent], List[float]]:
    """
    Compare different epsilons for epsilon-greedy algorithm.
    """
    agents = []
    bandits = [Bandit(m) for m in bandits_true_means]

    for epsilon in epsilons:
        logger.info("Running epsilon-greedy for epsilon = %f", epsilon)
        agent = EpsilonGreedyAgent(bandits=bandits, epsilon=epsilon)
        agent.take_actions(iterations)
        agents.append(agent)

    return agents, epsilons


epsilons = [0.01, 0.05, 0.1, None]
bandits_means = [3, 4, 5]

iterations = 50
agents, _ = compare_epsilons(epsilons, bandits_means, iterations)

all_rewards = [agent.rewards_log.all_rewards for agent in agents]

for i in range(len(agents)):
    plt.plot(
        np.cumsum(all_rewards[i]),
        label="epsilon = {}".format(epsilons[i] or '1 / iter'),
    )

plt.xlabel("iteration")
plt.ylabel("total rewards")
plt.legend()
plt.show()