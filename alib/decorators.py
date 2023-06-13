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
