from learnrl.environments.single_agents.minesweeper.game import MinesweeperEnv
from learnrl.environments.single_agents.minesweeper.agents import RandomAgent, OptimalAgent, HumanAgent
from learnrl import Playground

env = MinesweeperEnv(grid_shape=(16, 30), impact_size=3, chicken=True, bombs_density=0.15, n_bombs=None)
agent = OptimalAgent(env)
pg = Playground(env, agent)

pg.run(10, render=True, verbose=1)
