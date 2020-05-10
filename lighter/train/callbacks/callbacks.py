from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class Callback(object):
    """
    Abstract base class for all callbacks

    Lists all the basic functions required
    """
    def epoch_begin(self, cls):
        pass

    def __call__(self, report, cls, n):
        pass

    def epoch_end(self, cls):
        pass



class MovingAvgCallback(Callback):
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
    def __init__(self, stateful_metrics=None, **kwargs):
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()


    def epoch_begin(self, cls):
        self.avg = {}  # Dictionary of averages for the metrics and values
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

    Subclass of the moving average callback to make life easier

    Parameters
    ----------
    total: Integer or None
        The total length of the progress bar. If none, obtain it from the loader of the trainer.
        Note: if this is set to 0, then the tqdm progress bar won't actually have a bar, just the numbers.
    description: string
        Description of the progress bar
    check_queue: boolean
        Display whether the queue is empty for AsynchronousLoader
    """
    def __init__(self, total=None, description=None, check_queue=False, **kwargs):
        super(ProgBarCallback, self).__init__(**kwargs)
        self.total = total
        self.description = description
        self.check_queue = check_queue

    def epoch_begin(self, cls):
        super(ProgBarCallback, self).epoch_begin(cls)
        total = self.total if self.total is not None else len(cls.loader)
        self.pb = tqdm(total=total)
        self.pb.set_description(self.description)

    def __call__(self, report, cls, n):
        super(ProgBarCallback, self).__call__(report, cls, n)

        if self.check_queue:
            self.avg['queue_empty'] = cls.loader.queue.empty()

        self.pb.set_postfix(**self.avg)
        self.pb.update(1)

    def epoch_end(self, cls):
        super(ProgBarCallback, self).epoch_end(cls)
        self.pb.close()



class CheckpointCallback(MovingAvgCallback):
    """
    Basic callback to save the model every epoch

    Parameters
    ----------
    filename: string
        The location to save the model
    monitor: string
        The metric to monitor
    save_best: boolean
        Whether to save the best epoch model or just every epoch
    mode: string
        Either 'min' or 'max', so that we know whether the goal is to minimise or maximie the metrics when saving the best
    """

    def __init__(self, model, filename, monitor, save_best=False, mode='min', **kwargs):
        super(CheckpointCallback, self).__init__(**kwargs)
        self.model = model
        self.filename = filename
        self.monitor = monitor
        self.save_best = save_best
        self.mode = mode
        self.prev = None


    def epoch_begin(self, cls):
        super(CheckpointCallback, self).epoch_begin(cls)


    def epoch_end(self, cls):
        super(CheckpointCallback, self).epoch_end(cls)
        current = self.avg[self.monitor]
        if self.save_best:  # Only check if best if we want it to
            if self.prev is None:
                pass  # If this is the first epoch, ignore, we'll set prev to the current results later
            else:
                if self.mode == 'min':
                    if self.prev < current:
                        return  # If not minimum, then do nothing
                elif self.mode == 'max':
                    if self.prev > current:
                        return  # If not maximum, then do nothing
                else:
                    raise Exception(
                        'mode must be max or min, got {}'.format(self.mode))

        print('Best epoch so far with metric at {} beating previous best at {}, saving model.'.format(current, self.prev))
        self.prev = current
        device = next(self.model.parameters()).device
        torch.save(self.model.cpu().state_dict(), self.filename)
        self.model = self.model.to(device=device)
