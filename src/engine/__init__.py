"""Training/evaluation engine exports."""


def __getattr__(name):
    if name == "evaluate_run":
        from .evaluate import evaluate_run

        return evaluate_run
    if name == "train_and_eval":
        from .train import train_and_eval

        return train_and_eval
    raise AttributeError(name)


__all__ = ["evaluate_run", "train_and_eval"]
