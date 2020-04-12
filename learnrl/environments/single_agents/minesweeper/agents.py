from learnrl.core import Agent
from learnrl.environments import MinesweeperEnv

import numpy as np
import pygame

from copy import copy

class RandomAgent(Agent):

    def __init__(self, env:MinesweeperEnv):
        pass

    def act(self, observation:np.ndarray):
        grid_shape = observation.shape
        return (np.random.randint(grid_shape[0]), np.random.randint(grid_shape[1]), np.random.randint(2))
    
    def learn(self):
        pass

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass

class OptimalAgent(Agent):

    def __init__(self, env:MinesweeperEnv):
        self.impact_size = env.impact_size
        self.BOMB = env.BOMB
        self.HIDDEN = env.BOMB + 1
        self.FLAG = env.BOMB + 2
        self.clip_index = env.clip_index
        self.n_bombs = env.n_bombs
    

    def eval_around(self, observation, coords):
        x_min, x_max, _, _ = self.clip_index(coords[0], 0)
        y_min, y_max, _, _ = self.clip_index(coords[1], 1)
        region = copy(observation[x_min:x_max, y_min:y_max])
        revealed_coords = np.argwhere(region < self.BOMB)
        for revealed_coord in revealed_coords:
            revealed_coord += np.array([x_min, y_min])
            revealed_coord = tuple(revealed_coord)
            bombs_detected = observation[revealed_coord]

            bombs_flagged = np.sum(region == self.FLAG)
            hidden_blocks = np.sum(region == self.HIDDEN) - 1
            if bombs_detected == 1 and bombs_flagged == 0 and hidden_blocks == 1:
                return 1
            if hidden_blocks <= bombs_detected - bombs_flagged:
                return 1
            if bombs_detected == bombs_flagged:
                return 0
            print(revealed_coord, hidden_blocks, bombs_detected, bombs_flagged)
        return 2

    def act(self, observation:np.ndarray):
        if np.all(observation == self.HIDDEN):
            self.prior = np.ones_like(observation) * (self.n_bombs/(observation.size - self.n_bombs))
            return tuple(np.array(observation.shape)//2) + (0,)
        self.prior[observation < self.BOMB] = 0
        # print(self.prior)
        for index, state in np.ndenumerate(observation):
            if state < self.BOMB:
                x_min, x_max, _, _ = self.clip_index(index[0], 0)
                y_min, y_max, _, _ = self.clip_index(index[1], 1)
                region = copy(observation[x_min:x_max, y_min:y_max])
                if np.any(region == self.HIDDEN):
                    if np.sum(region == self.FLAG) == state:
                        return tuple(index) + (0,)
                    hidden_coords = np.argwhere(region == self.HIDDEN)
                    for hidden in hidden_coords:
                        hidden += np.array([x_min, y_min])
                        hidden = tuple(hidden)
                        print(index)
                        action = self.eval_around(observation, hidden)
                        if action == 0:
                            return hidden + (0,)
                        elif action == 1:
                            return hidden + (1,)
        return (np.random.randint(observation.shape[0]), np.random.randint(observation.shape[1]), 0)
    
    def learn(self):
        pass

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass

class HumanAgent(Agent):

    def __init__(self, env:MinesweeperEnv):
        self.env = env
 
    def act(self, observation:np.ndarray):
        waiting = True
        while waiting:
            if self.env.pygame_is_init:
                self.env.render()
                for event in pygame.event.get(): # pylint: disable=E1101
                    if event.type == pygame.QUIT: # pylint: disable=E1101
                        waiting = False
                    if event.type == pygame.MOUSEBUTTONDOWN: # pylint: disable=E1101
                        waiting = False
                        block_position = (np.array(event.pos) - self.env.origin) / (self.env.scale_factor * self.env.BLOCK_SIZE)
                        block_position = (int(block_position[1]), int(block_position[0]))
                        if event.button == 1: action = 0
                        elif event.button == 3: action = 1
                        else: waiting = True
            else:
                waiting = False
        return block_position + (action,)
    
    def learn(self):
        pass

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass
