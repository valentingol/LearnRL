# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import learnrl as rl
from learnrl.environments import CrossesAndNoughtsEnv, MinesweeperEnv
from learnrl.agents import TableAgent

env = MinesweeperEnv()
agent = TableAgent(observation_space=env.observation_space, action_space=env.action_space)

agents = [agent]
pg = rl.Playground(env, agents)
pg.fit(10000, verbose=1)
