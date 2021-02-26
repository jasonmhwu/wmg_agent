import numpy as np
from numpy.random import randint, choice, permutation
from typing import Tuple

from utils.spec_reader import spec

MIN_NUM_OBJECTS = spec.val("MIN_NUM_OBJECTS")
MAX_NUM_OBJECTS = spec.val("MAX_NUM_OBJECTS")
NUM_ACTIONS = spec.val("NUM_ACTIONS")
NUM_REPEATS = spec.val("NUM_REPEATS")
MAX_OBSERVATIONS = spec.val("MAX_OBSERVATIONS")


class Collins2018Task():
    """A task examining the interaction between human working memory and reinforcement learning.

    Input parameters:
    num_objects (int)
    available_range (tuple): the available range of integers the agent can select. (min, max)
    target (int): environment gives reward 1 if action == target
    obs_mode (str): if obs_mode is
        'all_history': the observation is a list recording all previous agent interactions,
            where each entry is either positive (1), negative (0), or unobserved (2).
        'prev_result': only the result (either 1 or -1) of the preceding action is given.
    """

    metadata = {'render.modes': ['console']}

    def __init__(self,
                 num_objects: Tuple[int, ...] = (3, 6),
                 num_actions: int = 3,
                 num_repeats: int = 13,
                 max_observations: int = 6,
                 mode: str = 'random'
                 ):
        """ Initializes task parameters.

        :param num_objects (tuple[int]): number of objects that can be specified.
        :param num_actions (int): number of available actions.
        """

        super().__init__()

        self.name = "Collins2018"
        self.num_objects = list(range(MIN_NUM_OBJECTS, MAX_NUM_OBJECTS+1))
        self.num_actions = NUM_ACTIONS
        self.num_repeats = NUM_REPEATS
        assert mode in ['random', 'permutation']
        self.mode = mode
        max_observations = max(MAX_OBSERVATIONS, max(self.num_objects))
        if self.mode == 'permutation':
            assert self.num_actions >= max_observations

        #self.observation_space = MultiDiscrete([
        #    max_observations,  # observe one of the objects each trial
        #    num_actions,  # previous action
        #    2  # reward from last action
        #])
        self.observation_space = 3

        #self.action_space = Discrete(self.num_actions)
        self.action_space = self.num_actions

        self.config = {
            'num_objects': self.num_objects,
            'num_actions': self.num_actions,
            'num_repeats': self.num_repeats
        }

        # Define state variables:
        self.curr_num_objects = choice(num_objects)
        self.target = randint(0, num_actions, size=self.curr_num_objects)
        self.curr_obs = randint(self.curr_num_objects)
        self.prev_obs = 0
        self.prev_action = 0  # now assumes that prevAct at t = 0 is 0?
        self.reward = 0
        self.t = 0
        self.done = False
        self.info = {
            'target': self.target,
            'curr_num_objects': self.curr_num_objects,
            'objects_iter': 0
        }
        self.total_steps = 0
        self.total_reward = 0
        self.reset_online_test_sums()

    def step(self, action):
        """Move the environment forward given the input action."""

        ans = self.target[self.curr_obs]

        self.reward = 1 if (action == ans) else 0
        self.done = True if self.t >= self.num_repeats*self.curr_num_objects - 1 else False
        self.prev_action = action
        self.info['objects_iter'] = self.t // self.curr_num_objects
        self.t += 1
        self.prev_obs = self.curr_obs
        self.curr_obs = randint(self.curr_num_objects)
        self.update_online_test_sums(self.reward, self.done)

        #return np.array([self.curr_obs, self.prev_action, self.reward]), self.reward, self.done, self.info
        return np.array([self.curr_obs, self.prev_action, self.reward]), self.reward, self.done

    def reset(self, repeat=False, episode_id = None):
        """Resets the environment variables."""
        self.curr_num_objects = choice(self.num_objects)
        # random initialize target
        if self.mode == 'random':
            self.target = randint(0, self.num_actions, size=self.curr_num_objects)
        elif self.mode == 'permutation':
            self.target = permutation(self.curr_num_objects)
        else:
            print('mode unrecognized.')
        self.info = {
            'target': self.target,
            'curr_num_objects': self.curr_num_objects,
            'objects_iter': 0
        }
        self.curr_obs = randint(self.curr_num_objects)
        self.prev_obs = 0
        # ideally, prev_action at t = 0 should not matter
        self.prev_action = randint(self.num_actions)
        self.reward = 0
        self.t = 0
        self.done = False

        # returns the first state
        return np.array([self.curr_obs, self.prev_action, self.reward])

    def render(self, mode='console'):
        """Display current status of the environment."""

        if mode != 'console':
            raise NotImplementedError('other render modes are not implemented yet')
        print('t = ', self.t)
        print('action is ', self.prev_action)
        print('observation is ', self.prev_obs)
        print('reward is ', self.reward)
        print('next state is ', np.array([self.curr_obs, self.prev_action, self.reward]))
        print('target in this episode is ', self.target)
        print('\n')

    def close(self):
        pass

    def reset_online_test_sums(self):
        # Called only by the environment itself.
        self.step_sum = 0
        self.reward_sum = 0.
        self.num_episodes = 0
        self.need_to_reset_sums = False

    def update_online_test_sums(self, reward, done):
        # Called only by the environment itself.
        if self.need_to_reset_sums:
            # Another thread recently called reduce_online_test_sums(), so the previous counts are stale.
            self.reset_online_test_sums()
        # If another thread happens to call reduce_online_test_sums near this point in time,
        # one sample from this agent might get dropped. But that's a small price to avoid locking.
        self.step_sum += 1
        self.reward_sum += reward
        self.total_steps += 1
        self.total_reward += reward
        if done:
            self.num_episodes += 1

    def report_online_test_metric(self):
        # Called by the reporting manager only.

        # Calculate the final metric for this test period.
        self.reward_percentage = 100.0 * self.reward_sum / self.step_sum  # Reward available on every other step.

        # Assemble the tuple to be returned.
        #   1. The number of steps in the period. (This will be a bit different for each running thread.)
        #   2. The actual metric value (must be negated if lower is better).
        #   3. A string containing the formatted metric.
        #   4. A string containing the metric's units for display.
        #ret = (self.step_sum, self.num_episodes, self.reward_percentage, "{:7.3f}".format(self.reward_percentage), "reward percentage", False)

        metrics = []
        metrics.append((self.reward_percentage, "{:7.3f}".format(self.reward_percentage), "% reward"))
        ret = (self.step_sum, self.num_episodes, self.reward_percentage, metrics, False)

        # Reset the global sums.
        self.reset_online_test_sums()

        return ret