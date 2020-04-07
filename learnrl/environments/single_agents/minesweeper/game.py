from gym import Env, spaces
from time import time
import numpy as np
from copy import copy

class MinesweeperEnv(Env):

    def __init__(self, grid_shape=(2, 2), bombs_density=0.2, n_bombs=None, impact_size=3, max_time=999):
        self.grid_shape = grid_shape
        self.grid_size = np.prod(grid_shape)
        self.n_bombs = int(bombs_density * self.grid_size) if n_bombs is None else n_bombs
        self.flaged_bombs = 0
        self.flaged_empty = 0
        self.max_time = max_time

        if impact_size % 2 == 0:
            raise ValueError('Impact_size must be an odd number !')
        self.impact_size = impact_size

        # Setting up gym Env conventions
        nvec_observation = (self.impact_size ** 2 + 2) * np.ones(self.grid_shape)
        self.observation_space = spaces.MultiDiscrete(nvec_observation)

        nvec_action = np.array(self.grid_shape + (2,))
        self.action_space = spaces.MultiDiscrete(nvec_action)

        # Initalize state
        self.state = np.zeros(self.grid_shape + (2,), dtype=np.uint8)

        ## Setup bombs places
        idx = np.indices(self.grid_shape).reshape(2, -1)
        bombs_ids = np.random.choice(range(self.grid_size), size=self.n_bombs, replace=False)
        self.bombs_positions = idx[0][bombs_ids], idx[1][bombs_ids]

        ## Place numbers
        semi_impact_size = (self.impact_size-1)//2
        def clip_index(x, axis):
            max_idx = self.grid_shape[axis]
            x_min, x_max = max(0, x-semi_impact_size), min(max_idx, x + semi_impact_size + 1)
            dx_min, dx_max = x_min - (x - semi_impact_size), x_max - (x + semi_impact_size + 1) + self.impact_size
            return x_min, x_max, dx_min, dx_max

        bomb_impact = np.ones((self.impact_size, self.impact_size), dtype=np.uint8)
        for bombs_id in bombs_ids:
            bomb_x, bomb_y = idx[0][bombs_id], idx[1][bombs_id]
            x_min, x_max, dx_min, dx_max = clip_index(bomb_x, 0)
            y_min, y_max, dy_min, dy_max = clip_index(bomb_y, 1)
            bomb_region = self.state[x_min:x_max, y_min:y_max, 0]
            bomb_region += bomb_impact[dx_min:dx_max, dy_min:dy_max]

        ## Place bombs
        self.state[self.bombs_positions + (0,)] = self.impact_size ** 2
        self.start_time = time()

    def get_observation(self):
        observation = copy(self.state[:, :, 1]) + self.impact_size ** 2 + 1

        revealed = observation == 1
        observation[revealed] = copy(self.state[:, :, 0][revealed])

        flaged = observation == 2
        observation[flaged] -= 1

    def reveal_around(self, coords):
        print('reveal')
        pass

    def step(self, action):
        coords = action[:2]

        action_type = action[2] + 1 # 0 -> 1 = reveal; 1 -> 2 = toggle_flag
        UNSEEN = 0
        REVEAL = 1

        case_state = self.state[coords + (1,)]
        FLAG = 2

        case_content = self.state[coords + (0,)]
        BOMB = self.impact_size ** 2

        reward, done = 0, False

        if self.start_time - time() > self.max_time:
            score = -(self.n_bombs - self.flaged_bombs + self.flaged_empty)/self.n_bombs
            reward, done = score, True
            print('You took too much time, the bombs have exploded !')
            return self.get_observation(), reward, done, {'passed':False}
        
        if action_type == REVEAL:
            if case_state == UNSEEN:
                if case_content == BOMB:
                    print('BOOM !!! You lost ...')
                    reward, done = -1, True
            elif case_state == REVEAL:
                self.reveal_around(coords)
                reward = -0.01
            else:
                reward = -0.001
                return self.get_observation(), reward, done, {'passed':True}
            
            self.state[coords + (1,)] = action_type

        elif action_type == FLAG:
            flaging = 1
            if case_state == FLAG:
                flaging = -1
                self.state[coords + (1,)] = UNSEEN
            else:
                self.state[coords + (1,)] = FLAG
            
            if case_content == BOMB:
                self.flaged_bombs += flaging
            else:
                self.flaged_empty += flaging

        if self.flaged_bombs == self.n_bombs:
            print('A winner is you !')
            reward, done = 1, True

        return self.get_observation(), reward, done, {'passed':False}

if __name__ == "__main__":
    env = MinesweeperEnv()
