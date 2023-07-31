# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from bandit_utilss import BernoulliBandit
from compare_agents import compare_agents
from utiuls import get_agents


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    probs = [0.6, 0.7, 0.8, 0.9]
    bernoulli_bandits = [BernoulliBandit(p) for p in probs]
    compare_agents(
        get_agents(),
        bernoulli_bandits,
        iterations=500,
        show_plot=True,
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
