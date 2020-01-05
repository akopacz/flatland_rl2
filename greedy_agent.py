import numpy as np
from flatland.envs.rail_env import RailEnvActions

class GreedyAgent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT,
                                 RailEnvActions.STOP_MOVING])
    
    def simple_act(self, p0, dir, p1):
        diff_0 = p1[0] - p0[0]
        diff_1 = p1[1] - p0[1]
        new_dir = 0
        if diff_0 < 0:
            # North
            new_dir = 0
        if diff_0 > 0:
            # SOUTH
            new_dir = 2
        if diff_1 > 0:
            # EAST
            new_dir = 1
        if diff_1 < 0:
            # West
            new_dir = 3
        # return action % 4
        # if abs(diff - dir) < 2:
        #     return 2 + diff - dir
        # else:
        #     return 4
        return self.change_dir_from_to(dir, new_dir)

    def change_dir_from_to(self, from_dir, to_dir):
        turn = to_dir - from_dir
        if turn > 1:
            turn -= 4
        elif turn < -1:
            turn += 4
        return 2 + turn

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return