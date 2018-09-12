from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm



class MovingAvgCallback(object):
    """
    Basic callback to keep a moving average of training/evaluation statistics
    This callback has no outputs, and is designed to be used by as a base class for any other moving average based callbacks

    Parameters
    ----------
    stateful_metrics: Iterable of string
        Iterable of string names of metrics that
        should *not* be averaged over time. Metrics in this list
        will be displayed as-is. All others will be averaged
        by the progbar before display.
    """
    def __init__(self, stateful_metrics = None, **kwargs):
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()


    def epoch_begin(self, cls):
        self.avg = {} # Dictionary of averages for the metrics and values
        self._accumulator = {}
        self._seen_so_far = 0


    def __call__(self, report, cls, n):
        values = dict(list(report.losses.items()) + list(report.metrics.items()))
        for k in values:
            if k not in self.stateful_metrics:
                if k not in self._accumulator:
                    self._accumulator[k] = [values[k] * (n - self._seen_so_far), n]
                else:
                    self._accumulator[k][0] += values[k] * (n - self._seen_so_far)
                    self._accumulator[k][1] = n
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the numeric formatting.
                self._accumulator[k] = [values[k], 1]

        self._seen_so_far = n

        for k in self._accumulator:
            self.avg[k] = self._accumulator[k][0] / max(1, self._accumulator[k][1])


    def epoch_end(self, cls):
        pass



class ProgBarCallback(MovingAvgCallback):
    """
    Basic callback to display a progress bar using tqdm

    Parameters
    ----------
    description: string
        Description of the progress bar
    check_queue: boolean
        Display whether the queue is empty for AsynchronousLoader
    """
    def __init__(self, description = None, check_queue = False, **kwargs):
        super(ProgBarCallback, self).__init__(**kwargs)
        self.description = description
        self.check_queue = check_queue


    def epoch_begin(self, cls):
        super(ProgBarCallback, self).epoch_begin(cls)
        self.pb = tqdm(total = len(cls.loader))
        self.pb.set_description(self.description)


    def __call__(self, report, cls, n):
        super(ProgBarCallback, self).__call__(report, cls, n)

        avg = {}
        for k in self.avg: # We need to convert them into NumPy first or tqdm will also display a load of extra information
            avg[k] = self.avg[k].detach().cpu().numpy()

        if self.check_queue:
            avg['queue_empty'] = cls.loader.queue.empty()

        self.pb.set_postfix(**avg)
        self.pb.update(1)


    def epoch_end(self, cls):
        super(ProgBarCallback, self).epoch_end(cls)
        self.pb.close()
