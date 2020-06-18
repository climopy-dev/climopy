#!/usr/bin/env python3
"""
Tools for manipulating docstrings.
"""
import inspect

#: Dictionary of docstring snippets added with `add_snippets`.
snippets = {}

def add_snippets(func):
    """Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names."""
    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= snippets
    return func
