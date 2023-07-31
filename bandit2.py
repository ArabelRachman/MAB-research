import logging
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from typing import List
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt


class NoBanditsError(Exception):
    ...


class Bandit:
    def __init__(self, m: float, lower_bound: float = None, upper_bound: float = None):
        """
        Simulates bandit.
        Args:
            m (float): True mean.
            lower_bound (float): Lower bound for rewards.
            upper_bound (float): Upper bound for rewards.
        """

        self.m = m
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.id = uuid4()

    def pull(self):
        """
        Simulate pulling the arm of the bandit.
        Normal distribution with mu = self.m and sigma = 1. If lower_bound or upper_bound are defined then the
        distribution will be truncated.
        """
        n = 10
        possible_rewards = np.random.randn(n) + self.m

        allowed = np.array([True] * n)
        if self.lower_bound is not None:
            allowed = possible_rewards >= self.lower_bound
        if self.upper_bound is not None:
            allowed *= possible_rewards <= self.upper_bound

        return possible_rewards[allowed][0]