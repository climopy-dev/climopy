#!/usr/bin/env python3
"""
Tools for manipulating docstrings.
"""
import inspect


class _SnippetManager(dict):
    """
    A simple database for handling documentation snippets.
    """
    def __call__(self, **kwargs):
        """
        Add snippets to the string or object using ``%(name)s`` substitution. Use
        `kwargs` to format the snippets themselves with ``%(name)s`` substitution.
        """
        kwargs = {key: string.strip() for key, string in kwargs.items()}
        def _decorator(obj):  # noqa: E306
            if isinstance(obj, str):
                obj %= self  # add snippets to a string
                obj %= kwargs
            else:
                obj.__doc__ = inspect.getdoc(obj)  # also dedents the docstring
                if obj.__doc__:
                    obj.__doc__ %= self  # insert snippets after dedent
                    obj.__doc__ %= kwargs
            return obj
        return _decorator

    def __setitem__(self, key, value):
        """
        Populate input strings with other snippets and strip newlines. Developers
        should take care to import modules in the correct order.
        """
        value = value.strip('\n')
        super().__setitem__(key, value)


# Initiate snippets database
_snippet_manager = _SnippetManager()
