#!/usr/bin/env python3
"""
Tools for manipulating docstrings.
"""
import inspect

#: The docstring snippets used with `inject_snippets`.
snippets = {}


def inject_snippets(**kwargs):
    """
    Return a decorator that dedents docstrings with `inspect.getdoc` and adds
    un-indented snippets from the `snippets` dictionary.

    Parameters
    ----------
    **kwargs
        Additional snippets applied after the global snippets. These can be used to
        format injected global snippets.

    Notes
    -----
    The oldschool notation ``'%(x)s' % {'x': 'foo'}`` is used rather than
    ``'{x}'.format(x='foo')`` to permit curly braces in the rest of the docstring.
    """
    def _decorator(func):
        doc = inspect.getdoc(func) or ''
        for kw in (snippets, kwargs):
            doc = doc % {k: s.strip() for k, s in kw.items()}
        func.__doc__ = doc.strip()
        return func
    return _decorator
