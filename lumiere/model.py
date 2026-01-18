from typing import Protocol


class Model(Protocol):
    """A machine learning model.

    This protocol defines methods that encapsulate fitting a model to some data and
    evaluating the models performance on some data.

    Both methods accept general args and kwargs to facilitate the implementation of a
    wide variety of tasks that a model may perform.
    """

    def fit(self, *arkgs, **kwargs):
        """Fit the model to some data."""
        pass

    def eval(self, *args, **kwargs):
        """Evaluate the models performance on some data."""
        pass
