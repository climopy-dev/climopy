#!/usr/bin/env python3
"""
Tools for manipulating docstrings.
"""
import inspect

#: Dictionary of docstring snippets added with `add_snippets`.
snippets = {}

#: Dictionary of docstring templates filled with `add_template`.
templates = {}


def add_snippets(func):
    """
    Decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the global `snippets` dictionary. This function
    uses ``%(name)s`` substitution rather than `str.format` substitution so
    that the `snippets` keys can be invalid variable names.
    """
    func.__doc__ = inspect.getdoc(func)
    if func.__doc__:
        func.__doc__ %= {key: value.strip() for key, value in snippets.items()}
    func.__doc__ = func.__doc__.strip()
    return func


def add_template(template, notes=None, **kwargs):
    """
    Return a decorator that replaces the docstring with the template. Optionally
    append notes to the end of the docstring with `notes`.
    """
    string = templates[template].format(**kwargs).strip()
    if notes:
        if isinstance(notes, str):
            notes = (notes,)
        string += '\n\nNotes\n-----\n'
        string += '\n\n'.join(snippets[s].strip() for s in notes)
    def _decorator(func):  # noqa: E306
        func.__doc__ = string
        return func
    return _decorator
