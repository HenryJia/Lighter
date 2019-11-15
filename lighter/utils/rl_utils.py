import random

class RingBuffer:
    """
    Basic ring buffer for storing replay memory

    Parameters
    ----------
    maxlen: Integer
        Maximum size of our ring
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = []
        self.idx = 0


    def push(self, experience):
        #experience = (state, action, reward, next_state)
        if len(self.data) < self.maxlen:
            self.data.append(experience) # Add to t he ring until it's full
        else:
            self.idx = self.idx % self.maxlen # Go around the ring
            self.data[self.idx] = experience
            self.idx += 1


    def sample(self, batch_size):
        return random.sample(self.data, batch_size)


    def __len__(self):
        return len(self.data)
