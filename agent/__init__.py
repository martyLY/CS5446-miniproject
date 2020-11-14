import random
# import expert_agent

try:
    from runner.abstracts import Agent
except:
    class Agent(object):
        pass

import random
import torch
import numpy as np

from enum import Enum


class ExpertAgent(Agent):

    def get_position(self, state, channel):
        # idx = torch.nonzero(state[channel])

        x, y = np.nonzero(state[channel, :, :])

        assert len(x) == 1
        return (x[0], y[0])

    def find_max_speed_no_crash(self, cars, lane, y):
        max_speed = -1

        # have empty space in front
        if y >= 1 and cars[lane, y - 1] == 0:
            max_speed -= 1
            # have empty space 2 spaces in front
            if y >= 2 and cars[lane, y - 2] == 0:
                max_speed -= 1

        # check if the agent is getting impossible regions by going forward too fast
        if y + max_speed < lane:
            max_speed = min(lane - y, -1)

        return max_speed

    def up_lane_danger_level(self, cars, lane, y):
        # assert lane > 0

        if cars[lane - 1, y]:
            return 1

        # if the right-back has a car, the mild risk of collision
        if cars[lane - 1, (y + 1) % 50]:
            return 0.8

        if cars[lane - 1, (y + 2) % 50]:#0.33
            return 0.2

        return 0

    # assess the risk of being tailgated and crash (getting too close)
    def tail_danger_level(self, cars, lane, y, speed):

        # not possible to be tail-gated
        if speed < -2:
            return 0

        if speed == -2:
            if cars[lane, (y + 1) % 50]:
                return 0.2
            else:
                # not possible to be tail-gated if there is no car directly behind
                return 0

        # got car tail gating
        if cars[lane, (y + 1) % 50]:
            return 0.8

        return 0

    '''
    An example agent that just output a random action.
    '''

    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.

        For example, you might want to load the appropriate neural networks weight
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')

        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        # fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range = kwargs.get('agent_speed_range')
        # gamma               = kwargs.get('gamma')
        self.width = kwargs.get('width')
        self.lanes = kwargs.get('lanes')

        # since agent_speed_range is given, we can fix the actions
        self.UP = 0
        self.DOWN = 1
        # this is forward[-1] position
        self.FORWARD = 2 + (agent_speed_range[1] - agent_speed_range[0] + 1)

        self.cars = 0
        self.agent = 1
        self.finish_position = 2
        self.occupancy_trails = 3

        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def reset(self, state, *args, **kwargs):
        '''
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        pass

    def step(self, state, *args, **kwargs):

        '''
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''

        # Uncomment to help debugging
        # print('>>> STEP >>>')
        # print('state:', state.shape)

        #state = int(state)

        agent_lane, agent_y = self.get_position(state, self.agent)

        # print(agent_lane, agent_y)

        # calculate maximum alloable forward speed, without crashing for sure (but may be crashed from behind though)
        max_speed = self.find_max_speed_no_crash(state[self.cars], agent_lane, agent_y)

        # assume the goal is always on the top row, then we are set to just go forward
        if agent_lane == 0:
            return int(self.FORWARD + max_speed)

        # decide if we should move up
        up_danger = self.up_lane_danger_level(state[self.cars], agent_lane, agent_y)

        speeds = np.arange(-1, max_speed - 1, -1, dtype=int)
        np.random.shuffle(speeds)

        # print(speeds)

        for speed in speeds:
            tail_danger = self.tail_danger_level(state[self.cars], agent_lane, agent_y, speed)

            if (tail_danger < up_danger):
                return int(self.FORWARD + speed)

        return int(self.UP)

    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.

        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state = kwargs.get('state')
        action = kwargs.get('action')
        reward = kwargs.get('reward')
        next_state = kwargs.get('next_state')
        done = kwargs.get('done')
        info = kwargs.get('info')

        # Uncomment to help debugging
        # print('>>> UPDATE >>>')
        # print('state:', state)
        # print('action:', action)
        # print('reward:', reward)
        # print('next_state:', next_state)
        # print('done:', done)
        # print('info:', info)

'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
# try: # server-compatible import statement
#     from models import *
# except: pass
# try: # local-compatible import statement
#     from .models import *
# except: pass





def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return ExpertAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_random_lane_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    # def test(agent, env, runs=1000, t_max=100):
    #     rewards = []
    #     agent_init = {'agent_speed_range': (-3,-1), 'width': env.width, 'lanes': len(env.lanes)}
    #     agent.initialize(**agent_init)
    #     for run in range(runs):
    #         state = env.reset()
    #         agent.reset(state)
    #         episode_rewards = 0.0
    #         for t in range(t_max):
    #             action = agent.step(state)
    #             action = int(action)
    #             # print(action)
    #             # env.render()
    #             next_state, reward, done, info = env.step(action)
    #             full_state = {
    #                 'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
    #                 'done': done, 'info': info
    #             }
    #             agent.update(**full_state)
    #             state = next_state
    #             episode_rewards += reward
    #             if done:
    #                 break
    #         rewards.append(episode_rewards)
    #         # input()
    #     avg_rewards = sum(rewards)/len(rewards)
    #     print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
    #     return avg_rewards
    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3, -1), 'gamma': 1}
        agent.initialize(**agent_init)
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards) / len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            # hm: tc['env'] stores the gym env object
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600, # 600 
            'testcases': [{ 'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs] # 300
        }

    task = get_task()
    timed_test(task)
