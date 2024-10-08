import numpy as np

class Schedule:
    def __init__(self):
        pass

    def value(self, episode):
        return 0.0

class LinearSchedule(Schedule):
    def __init__(self, start, end, num_episodes):
        super().__init__()
        self._start = start
        self._end = end
        self._step = (self._end - self._start) / num_episodes  # pls note: this can be negative, by intention, if end < start!

    def value(self, episode):
        v = self._start + self._step * episode

        if self._end > self._start: 
            return max(self._start, min(self._end, v))
        else:
            return min(self._start, max(self._end, v))
        
class ModuloWrapperSchedule(Schedule):
    """
    Wraps a schedule and returns a default value for every n-th episode, and
    the wrapped schedule's value for the other episodes. Can be used for
    having a greedy evaluation every nth episode, for instance. With
    negate=True, the behavior is inverted, thus returning the wrapped 
    schedule's value only for every n-th episode.
    """
    def __init__(self, schedule, modulo, default_value=0.0, negate=False):
        super().__init__()
        self._schedule = schedule
        self._modulo = modulo
        self._default_value = default_value
        self._negate = negate

    def value(self, episode):
        if self._negate:
            return self._default_value if episode % self._modulo == 0 else self._schedule.value(episode)
        else:
            return self._schedule.value(episode) if episode % self._modulo != 0 else self._default_value
