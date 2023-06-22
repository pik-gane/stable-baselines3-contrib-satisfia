def interpolate(lower_bound, t, upper_bound):
    """
    Linear interpolation between lower_bound and upper_bound

    :param lower_bound: lower bound
    :param t: interpolation factor
    :param upper_bound: upper bound
    """
    return lower_bound + (upper_bound - lower_bound) * t


def ratio(lower_bound, value, upper_bound):
    """
    Returns the ratio of value between lower_bound and upper_bound.
    If value is lower_bound, this returns 0. If value is upper_bound, this
    returns 1.

    :param lower_bound: The lower bound of the range.
    :param value: The value to be scaled.
    :param upper_bound: The upper bound of the range.
    """
    return (value - lower_bound) / (upper_bound - lower_bound)
