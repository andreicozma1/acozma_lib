from datetime import timedelta
from timeit import default_timer as timer


def listify(func):
    """
    This decorator allows a function to be applied to a list of items and returns a list accordingly.
    TODO: only works for functions that take one argument for now
    """

    def wrapper(arg, *args, **kwargs):
        # return [func(x) for x in arg] if isinstance(arg, list) else func(arg)
        results = []
        for x in arg:
            results.append(func(x, *args, **kwargs))
        return results

    return wrapper


def time_this(func):
    # using timeit will produce far more accurate results since it will automatically account for things like garbage collection and OS differences
    def wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(f"Elapsed time: {timedelta(seconds=end-start)}")
        return result

    return wrapper
