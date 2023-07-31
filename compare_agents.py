import numpy as np
from matplotlib import pyplot as plt
from oosapy.models import List

from agent2 import Agent
from bandit2 import Bandit
from bandit_utilss import logger


def compare_agents(agents: List[Agent], bandits: List[Bandit], iterations: int, show_plot=True):
    for agent in agents:
        logger.info("Running for agent = %s", agent)
        agent.bandits = bandits
        agent.take_actions(iterations)
        if show_plot:
            plt.plot(np.cumsum(agent.rewards_log.all_rewards), label=str(agent))

    if show_plot:
        plt.xlabel("iteration")
        plt.ylabel("total rewards")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()