"""Shared input checks for Python, HTTP and browser environments."""

import numpy as np


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a whole number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return int(value)


def seed_value(seed):
    return None if seed is None else integer(seed, "Seed")
