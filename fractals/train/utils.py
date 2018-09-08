from collections import OrderedDict

from queue import Queue
from threading import Thread
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

def load_loop(q, loader):
    for i, (data, targets) in enumerate(loader):
        data = [d.cuda(non_blocking = True) for d in data]
        targets = [t.cuda(non_blocking = True) for t in targets]
        q.put((data, targets))
        q.task_done()


def make_queue(loader, maxsize = 10):
    data_queue = Queue(maxsize = maxsize)
    worker = Thread(target = load_loop, args=(data_queue, loader))
    worker.setDaemon(True)
    worker.start()

    return data_queue, worker



class ProgressBar(tqdm):
    """
    Displays a progress bar based on combination of tqdm code and Keras' prgress bar code
    Parameters
    ----------
    total: int or None
        Total number of steps expected, None if unknown.
    stateful_metrics: Iterable of string
        Iterable of string names of metrics that
        should *not* be averaged over time. Metrics in this list
        will be displayed as-is. All others will be averaged
        by the progbar before display.
    """
    def __init__(self, total = None, description = None, stateful_metrics = None, **kwargs):
        super(ProgressBar, self).__init__(total = total, **kwargs)
        if description:
            self.set_description(description)

        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._avg = OrderedDict() # Dictionary of averages for the metrics and values
        self._values = OrderedDict()

    def update(self, n, values):
        """
        Updates the progress bar.
        Parameters
        ----------
        n: integer
            Amount to increase the progress bar by
        values: Dictionary
            Dictionary of `{name : value}`.
            If `name` is in `stateful_metrics`, `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
        """
        super(ProgressBar, self).update(n)
        values = values or []
        for k in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * n, n]
                else:
                    self._values[k][0] += v * n
                    self._values[k][1] += n
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]

        for k in self._values:
            self._avg = self._values[k][0] / max(1, self._values[k][1])

        pb.set_postfix(**self.avg)

    def print_final(self): # Print the final result
        result_str = 'Results: '
        for k in self.


class Accuracy(nn.Module):
    def __init__(self, dim = 1):
        super(Accuracy, self).__init__()
        self.dim = dim

    def forward(self, out, target):
        return torch.mean((torch.max(out, dim = self.dim)[1] == target).float())



class RMSELoss(MSELoss):
    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        return torch.sqrt(super(RMSELoss, self).forward(input, target))
