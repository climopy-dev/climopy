#!/usr/bin/env python3
"""
Warnings used internally by climo.
"""
import functools
import re
import sys
import warnings

import pint
import xarray as xr

# Store methods
catch_warnings = warnings.catch_warnings  # avoid overwriting module name
simplefilter = warnings.simplefilter  # avoid overwriting module name

# Filter warnings
simplefilter('error', category=pint.UnitStrippedWarning)
simplefilter('ignore', category=xr.core.extensions.AccessorRegistrationWarning)

# Warning class
ClimoPyWarning = type('ClimoPyWarning', (UserWarning,), {})


def _warn_climopy(message):
    """
    Emit a `ClimoPyWarning` and try to show the stack level corresponding
    to user code by jumping to the stack outside of climopy, numpy, scipy,
    pandas, xarray, pint, and scipy.
    """
    frame = sys._getframe()
    stacklevel = 1
    while True:
        if frame is None:
            break  # when called in embedded context may hit frame is None
        if not re.match(
            r'\A(climopy|numpy|scipy|xarray|pandas|pint).',
            frame.f_globals.get('__name__', '')
        ):
            break
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, ClimoPyWarning, stacklevel=stacklevel)


def _rename_objs(version, **kwargs):
    """
    Emit a basic deprecation warning after renaming function(s), method(s), or
    class(es). Each key should be an old name, and each argument should be the new
    object to point to. Do not document the deprecated object(s) to discourage use.
    """
    wrappers = []
    for old_name, new_obj in kwargs.items():
        new_name = new_obj.__name__
        message = (
            f'{old_name!r} was deprecated in version {version} and will be '
            f'removed in a future release. Please use {new_name!r} instead.'
        )
        if isinstance(new_obj, type):
            class _deprecate_obj(new_obj):
                def __init__(self, *args, new_obj=new_obj, message=message, **kwargs):
                    _warn_climopy(message)
                    super().__init__(*args, **kwargs)
        elif callable(new_obj):
            def _deprecate_obj(*args, new_obj=new_obj, message=message, **kwargs):
                _warn_climopy(message)
                return new_obj(*args, **kwargs)
        else:
            raise ValueError(f'Invalid deprecated object replacement {new_obj!r}.')
        _deprecate_obj.__name__ = old_name
        wrappers.append(_deprecate_obj)
    if len(wrappers) == 1:
        return wrappers[0]
    else:
        return tuple(wrappers)


def _rename_kwargs(version, **kwargs_rename):
    """
    Emit a basic deprecation warning after removing or renaming keyword argument(s).
    Each key should be an old keyword, and each argument should be the new keyword
    or *instructions* for what to use instead.
    """
    def decorator(func_orig):
        @functools.wraps(func_orig)
        def _deprecate_kwargs(*args, **kwargs):
            for key_old, key_new in kwargs_rename.items():
                if key_old not in kwargs:
                    continue
                value = kwargs.pop(key_old)
                if key_new.isidentifier():
                    # Rename argument
                    kwargs[key_new] = value
                elif '{}' in key_new:
                    # Nice warning message, but user's desired behavior fails
                    key_new = key_new.format(value)
                _warn_climopy(
                    f'Keyword {key_old!r} was deprecated in version {version} and will '
                    f'be removed in a future release. Please use {key_new!r} instead.'
                )
            return func_orig(*args, **kwargs)
        return _deprecate_kwargs
    return decorator
