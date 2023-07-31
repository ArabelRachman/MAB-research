from uuid import uuid4

import numpy as np


class Bandit:
    def __init__(self, m):
        """
        Simulates bandit.
        Args:
            m (float): True mean.
        """

        self.m = m
        self.id = uuid4()

    def pull(self):
        """
        Simulate pulling the arm of the bandit.
        Normal distribution with mu = self.m and sigma = 1.
        """
        reward = np.random.randn() + self.m

        return reward