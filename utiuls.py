import logging
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from samples.rl import ucb
from samples.rl.bandit import (
    Agent,
    Bandit,
    BernoulliBandit,
)
from samples.rl.epsilon_greedy import EpsilonGreedyAgent
from samples.rl.optimistic_initial_values import OptimisticInitialValuesAgent
from samples.rl.thompson_sampling import BayesianAgent

logger = logging.getLogger(__name__)


def compare_agents(
        agents: List[Agent],
        bandits: List[Bandit],
        iterations: int,
        show_plot=True,
):
    for agent in agents:
        logger.info("Running for agent = %s", agent)
        agent.bandits = bandits
        if isinstance(agent, ucb.UCBAgent):
            agent.initialise()

        N = iterations - agent.rewards_log.total_actions
        agent.take_actions(N)
        if show_plot:
            cumulative_rewards = np.cumsum(
                agent.rewards_log.all_rewards,
            )
            plt.plot(cumulative_rewards, label=str(agent))

    if show_plot:
        plt.xlabel("iteration")
        plt.ylabel("total rewards")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def get_agents():
    agents = [
        EpsilonGreedyAgent(),
        ucb.UCB1Agent(),
        ucb.UCB1TunedAgent(),
        BayesianAgent('bernoulli'),
    ]
    return agents


def run_comparison(bandits):
    win_count = [0] * len(get_agents())

    for _ in range(1000):
        agents = get_agents()
        iterations = 1000
        compare_agents(agents, bandits, iterations, show_plot=False)

        rewards = [agent.rewards_log.total_rewards for agent in agents]
        win_count[np.argmax(rewards)] += 1

    return win_count