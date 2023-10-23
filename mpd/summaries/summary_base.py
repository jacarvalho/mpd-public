import abc


class SummaryBase:

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def summary_fn(self, *args, **kwargs):
        pass
