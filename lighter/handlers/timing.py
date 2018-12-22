from ignite.engine import Events
from ignite.handlers import Timer


class EpochTimer(Timer):
    '''
        Simple wrapper for the ignite Timer to act as an epoch timer and save us from having to remember all the arguments
    '''
    def __init__(self, average = False, **kwargs):
        super(EpochTimer, self).__init__(average)

    def attach(self, engine, **kwargs):
        return super(EpochTimer, self).attach(engine, start = Events.EPOCH_STARTED, resume = Events.ITERATION_STARTED, pause = Events.ITERATION_COMPLETED, step = Events.ITERATION_COMPLETED)
