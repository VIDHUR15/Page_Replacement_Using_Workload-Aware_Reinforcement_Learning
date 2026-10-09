"""
Tabular Q-learning agent for policy selection.

Treated as a contextual-bandit-flavoured RL problem: state = discretized
workload features for the current window, action = which classic policy to
apply for the *next* window, reward = hit rate achieved on that window.
A full Q-update (with a next-state max term) is kept so the agent can still
credit a state for leading into windows that stay easy to serve, rather than
only judging the immediate window in isolation.
"""

import random
from collections import defaultdict

ACTIONS = ["FIFO", "LRU", "LFU", "MRU"]


class QLearningPolicySelector:
    def __init__(self, actions=ACTIONS, alpha=0.2, gamma=0.5,
                 epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.02, seed=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = random.Random(seed)
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def select_action(self, state: tuple, greedy: bool = False) -> str:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        qvals = self.Q[state]
        best = max(qvals.values())
        # break ties randomly so the agent doesn't get stuck on one action
        best_actions = [a for a, v in qvals.items() if v == best]
        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state) -> None:
        best_next = max(self.Q[next_state].values())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
