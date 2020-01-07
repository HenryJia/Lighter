from collections import namedtuple



StepReport = namedtuple('StepReport', ['outputs', 'losses', 'metrics'])



class RLStep(object):
    """
    Base class for a reinforcement learning step

    """
    def episode_begin(self):
        pass


    def __call__(self):
        raise NotImplementedError


    def episode_end(self):
        pass
