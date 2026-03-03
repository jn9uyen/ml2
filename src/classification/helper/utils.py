import inspect


def filter_kwargs(func, kwargs):
    """
    Filters kwargs to include only those that are accepted by the function.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
