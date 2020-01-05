import random

class RingBuffer(list):
    """
    Basic ring buffer for storing replay memory, subclass of list to make life easy

    Parameters
    ----------
    maxlen: Integer
        Maximum size of our ring
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.idx = 0


    def push(self, experience):
        #experience = (state, action, reward, next_state)
        if len(self) < self.maxlen:
            self.append(experience) # Add to t he ring until it's full
        else:
            self.idx = self.idx % self.maxlen # Go around the ring
            self[self.idx] = experience
            self.idx += 1


    def sample(self, batch_size):
        return random.sample(self, batch_size)
