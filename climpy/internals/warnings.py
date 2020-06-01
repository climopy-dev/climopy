#!/usr/bin/env python3
"""
Warnings used internally by climpy.
"""
import warnings


def _warn_climpy(message):
    """
    Emit a basic warning. This will be further developed in the future.
    """
    warnings.warn(message, stacklevel=3)  # 2, plus get out of this function
