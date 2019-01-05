from collections import deque
import random
from utils.utilities import partial_obs_2_full_state


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        # input_to_buffer = partial_obs_2_full_state(transition)
        #
        # for item in input_to_buffer:
        #     self.deque.append(item)

        self.deque.append(transition)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        # return partial_obs_2_full_state(samples)
        return samples

    def __len__(self):
        return len(self.deque)



