# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
from copy import copy
import warnings

from time import time
from gym import Env

from learnrl import Agent, TurnEnv
from learnrl.callbacks import CallbackList, Logger


class Playground():

    """ A playground is used to run agent(s) on an environement.

    Attributes
    ----------
        env: :class:`TurnEnv` or |gym.Env|
            Environement in which agents will play
        agents: list or :class:`Agent`
            List of :class:`Agent` to run on the :class:`TurnEnv`

    """

    def __init__(self, environement:Env, agents:Agent):
        assert isinstance(environement, Env)
        if isinstance(agents, Agent):
            agents = [agents]
        for agent in agents:
            assert isinstance(agent, Agent)

        self.env = environement
        self.agents = agents

    def run(self, episodes, render=True, learn=True, steps_cycle_len=10, episodes_cycle_len=0.05,
                  verbose=0, callbacks=[], logger=None, reward_handler=None, done_handler=None, **kwargs):
        
        """ Let the agent(s) play on the environement for a number of episodes.

        Additional arguments will be passed to the default logger if none is given, see :class:`~learnrl.callbacks.logger.Logger`.
        
        Parameters
        ----------
            episodes: int
                Number of episodes to run.
            render: bool
                If True, call :meth:`TurnEnv.render` every step.
            learn: bool
                If True, call :meth:`Agent.learn` every step.
            steps_cycle_len: int
                Number of steps that compose a cycle.
            episode_cycle_len: int or float
                Number of episodes that compose a cycle. If between 0 and 1, this in understood as a proportion.
            verbose: int
                The verbosity level: 0 (silent), 1 (cycle), 2 (episode), 3 (step_cycle), 4 (step), 5 (detailed step).
            callbacks: list
                List of :class:`~learnrl.callbacks.Callback` to use in runs.
            reward_handler: func or :class:`RewardHandler`
                A callable to redifine rewards of the environement.
            done_handler: func or :class:`DoneHandler`
                A callable to redifine the environement end.
            logger: :class:`~learnrl.callbacks.logging_callback.LoggingCallback`
                Logging callback to use, if none use the default :class:`~learnrl.callbacks.logger.Logger`.
        
        """
        if episodes_cycle_len < 1 and episodes_cycle_len > 0:
            episodes_cycle_len = max(1, int(episodes_cycle_len*episodes))
        else:
            assert episodes_cycle_len > 0, 'episodes_cycle_len must be > 0'
            episodes_cycle_len = int(episodes_cycle_len)

        params = {
            'episodes': episodes,
            'episodes_cycle_len': episodes_cycle_len,
            'steps_cycle_len': steps_cycle_len,
            'verbose': verbose,
            'render': render,
            'learn': learn
        }

        logger = logger if logger else Logger(**kwargs)
        callbacks = CallbackList(callbacks + [logger])
        callbacks.set_params(params)
        callbacks.set_playground(self)

        logs = {}
        logs.update(params)
        callbacks.on_run_begin(logs)

        for episode in range(episodes):

            if episode % episodes_cycle_len == 0:
                callbacks.on_episodes_cycle_begin(episode, logs)

            observation = self.env.reset()
            if isinstance(reward_handler, RewardHandler):
                reward_handler.reset()
            if isinstance(done_handler, DoneHandler):
                done_handler.reset()

            previous = [{'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None} for _ in range(len(self.agents))]
            done = False
            step = 0

            logs.update({'episode':episode})
            callbacks.on_episode_begin(episode, logs)

            while not done:

                if render: self.env.render()

                agent_id = self.env.turn(observation) if isinstance(self.env, TurnEnv) else 0
                if agent_id >= len(previous):
                    raise ValueError(f'Not enough agents to play environement {self.env}')
                agent = self.agents[agent_id]

                prev = previous[agent_id]
                if learn and prev['observation'] is not None:
                    agent.remember(prev['observation'], prev['action'], prev['reward'], prev['done'], observation, prev['info'])
                    agent_logs = agent.learn()
                    logs.update({f'agent_{agent_id}': agent_logs})

                logs.update({'step':step, 'agent_id':agent_id, 'observation':observation})

                if step % steps_cycle_len == 0:
                    callbacks.on_steps_cycle_begin(step, logs)

                callbacks.on_step_begin(step, logs)

                action = agent.act(observation, greedy=not learn)
                next_observation, reward, done, info = self.env.step(action)

                logs.update({'reward': reward})
                if reward_handler is not None:
                    reward = reward_handler(observation, action, reward, done, info, next_observation)
                    logs.update({'handled_reward': reward})

                logs.update({'done': done})
                if done_handler is not None:
                    done = done_handler(observation, action, reward, done, info, next_observation)
                    logs.update({'handled_done': done})

                if learn:
                    for key, value in zip(prev, [observation, action, reward, done, info]):
                        prev[key] = value
                    if done:
                        agent.remember(observation, action, reward, done, next_observation, info)
                        agent_logs = agent.learn()
                        logs.update({f'agent_{agent_id}': agent_logs})

                logs.update({'action': action, 'next_observation': next_observation})
                logs.update(info)

                callbacks.on_step_end(step, logs)

                step += 1
                observation = next_observation

                if (step + 1) % steps_cycle_len == 0 or done:
                    callbacks.on_steps_cycle_end(step, logs)

            callbacks.on_episode_end(episode, logs)

            if (episode + 1) % episodes_cycle_len == 0 or episode == episodes - 1:
                callbacks.on_episodes_cycle_end(episode, logs)

        callbacks.on_run_end(logs)

    def fit(self, episodes, **kwargs):
        """Train the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop('learn', True)
        render = kwargs.pop('render', False)
        if not learn:
            warnings.warn("learn should be True in Playground.fit(), otherwise the agents will not improve", UserWarning)
        if render:
            warnings.warn("rendering degrades heavily computation speed", RuntimeWarning)

        self.run(episodes, render=render, learn=learn, **kwargs)

    def test(self, episodes, **kwargs):
        """Test the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop('learn', False)
        render = kwargs.pop('render', True)
        verbose = kwargs.pop('verbose', 0)
        if learn:
            warnings.warn("learn should be False in Playground.test(), otherwise the agents will not act greedy and can have random behavior", UserWarning)
        if not render and verbose == 0:
            warnings.warn("you should set verbose > 0 or render=True to have any feedback ...", UserWarning)
        self.run(episodes, render=render, learn=learn, verbose=verbose, **kwargs)
    

class DoneHandler():

    """ Helper to modify the done given by the environmen

    You need to specify the method:
     - `done(self, observation, action, reward, done, info, next_observation) -> done`
    
    You can also define __init__ and reset() if you want to store anything.

    """

    def done(self, observation, action, reward, done, info, next_observation):
        raise NotImplementedError

    def reset(self):
        pass

    def _done(self, *args):
        done = self.done(*args)
        if not isinstance(done, (bool, np.bool, np.bool_)):
            raise ValueError(f"Done should be bool, got {done} of type {type(done)} instead")
        return done
    
    def __call__(self, *args):
        return self._done(*args)

class RewardHandler():

    """ Helper to modify the rewards given by the environment
    
    You need to specify the method:
     - `reward(self, observation, action, reward, done, info, next_observation) -> reward`
    
    You can also define __init__ and reset() if you want to store anything.
    
    """

    def reward(self, observation, action, reward, done, info, next_observation):
        raise NotImplementedError

    def reset(self):
        pass

    def _reward(self, *args):
        reward = self.reward(*args)
        if not isinstance(reward, (int, float)):
            raise ValueError(f"Rewards should be scalars, got {reward} of type {type(reward)} instead")
        return reward
    
    def __call__(self, *args):
        return self._reward(*args)
