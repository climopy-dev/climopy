#!/usr/bin/env python3
"""
Define a `Numeric` class for simple array-like wrappers.
"""
import copy
import functools
import numbers


class Quantity(object):
    """
    Mixin class for thin wrappers around numeric types whose numeric data
    is stored on the ``data`` attribute. This preserves metadata for natural
    operations. Wrappers must be applied manually, i.e. we cannot simply
    override `__getattr__`. See `this page \
<https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_.
    """
    # TODO: Better understanding of mixin classes? Maybe can do this with
    # an abstract base class? Probably very inefficient!
    # Representation
    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        return str(self.data)

    # Boolean
    def __bool__(self):
        return bool(self.data)

    # Container like properties
    def __len__(self):
        return len(self.data)

    def __contains__(self, other):
        return other in self.data

    def __iter__(self):
        if hasattr(self.data, '__iter__'):
            return self.data.__iter__()
        else:
            raise TypeError

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __delitem__(self, key):
        self.data.__delitem__(key)

    # Standard operations that accept other args, e.g. +
    def __add__(self, other, *args):
        return self._math_other('__add__', other, *args)

    def __sub__(self, other, *args):
        return self._math_other('__sub__', other, *args)

    def __mul__(self, other, *args):
        return self._math_other('__mul__', other, *args)

    def __matmul__(self, other, *args):
        return self._math_other('__matmul__', other, *args)

    def __truediv__(self, other, *args):
        return self._math_other('__truediv__', other, *args)

    def __floordiv__(self, other, *args):
        return self._math_other('__floordiv__', other, *args)

    def __mod__(self, other, *args):
        return self._math_other('__mod__', other, *args)

    def __divmod__(self, other, *args):
        return self._math_other('__divmod__', other, *args)

    def __pow__(self, other, *args):
        return self._math_other('__pow__', other, *args)

    def __lshift__(self, other, *args):
        return self._math_other('__lshift__', other, *args)

    def __rshift__(self, other, *args):
        return self._math_other('__rshift__', other, *args)

    def __and__(self, other, *args):
        return self._math_other('__and__', other, *args)

    def __xor__(self, other, *args):
        return self._math_other('__xor__', other, *args)

    def __or__(self, other, *args):
        return self._math_other('__or__', other, *args)

    def __radd__(self, other, *args):
        return self._math_other('__radd__', other, *args)

    def __rsub__(self, other, *args):
        return self._math_other('__rsub__', other, *args)

    def __rmul__(self, other, *args):
        return self._math_other('__rmul__', other, *args)

    def __rmatmul__(self, other, *args):
        return self._math_other('__rmatmul__', other, *args)

    def __rtruediv__(self, other, *args):
        return self._math_other('__rtruediv__', other, *args)

    def __rfloordiv__(self, other, *args):
        return self._math_other('__rfloordiv__', other, *args)

    def __rmod__(self, other, *args):
        return self._math_other('__rmod__', other, *args)

    def __rdivmod__(self, other, *args):
        return self._math_other('__rdivmod__', other, *args)

    def __rpow__(self, other, *args):
        return self._math_other('__rpow__', other, *args)

    def __rlshift__(self, other, *args):
        return self._math_other('__rlshift__', other, *args)

    def __rrshift__(self, other, *args):
        return self._math_other('__rrshift__', other, *args)

    def __rand__(self, other, *args):
        return self._math_other('__rand__', other, *args)

    def __rxor__(self, other, *args):
        return self._math_other('__rxor__', other, *args)

    def __ror__(self, other, *args):
        return self._math_other('__ror__', other, *args)

    def __lt__(self, other, *args):
        return self._math_other('__lt__', other, *args)

    def __le__(self, other, *args):
        return self._math_other('__le__', other, *args)

    def __eq__(self, other, *args):
        return self._math_other('__eq__', other, *args)

    def __ne__(self, other, *args):
        return self._math_other('__ne__', other, *args)

    def __gt__(self, other, *args):
        return self._math_other('__gt__', other, *args)

    def __ge__(self, other, *args):
        return self._math_other('__ge__', other, *args)

    # Inplace operations that accept other arg, e.g. +=
    def __iadd__(self, other, *args):
        return self._math_inplace('__iadd__', other, *args)

    def __isub__(self, other, *args):
        return self._math_inplace('__isub__', other, *args)

    def __imul__(self, other, *args):
        return self._math_inplace('__imul__', other, *args)

    def __imatmul__(self, other, *args):
        return self._math_inplace('__imatmul__', other, *args)

    def __itruediv__(self, other, *args):
        return self._math_inplace('__itruediv__', other, *args)

    def __ifloordiv__(self, other, *args):
        return self._math_inplace('__ifloordiv__', other, *args)

    def __imod__(self, other, *args):
        return self._math_inplace('__imod__', other, *args)

    def __ipow__(self, other, *args):
        return self._math_inplace('__ipow__', other, *args)

    def __ilshift__(self, other, *args):
        return self._math_inplace('__ilshift__', other, *args)

    def __irshift__(self, other, *args):
        return self._math_inplace('__irshift__', other, *args)

    def __iand__(self, other, *args):
        return self._math_inplace('__iand__', other, *args)

    def __ixor__(self, other, *args):
        return self._math_inplace('__ixor__', other, *args)

    def __ior__(self, other, *args):
        return self._math_inplace('__ior__', other, *args)

    # Operations that accept one arg, e.g. abs()
    def __neg__(self, *args):
        return self._math_unary('__neg__', *args)

    def __pos__(self, *args):
        return self._math_unary('__pos__', *args)

    def __abs__(self, *args):
        return self._math_unary('__abs__', *args)

    def __invert__(self, *args):
        return self._math_unary('__invert__', *args)

    def __complex__(self, *args):
        return self._math_unary('__complex__', *args)

    def __int__(self, *args):
        return self._math_unary('__int__', *args)

    def __float__(self, *args):
        return self._math_unary('__float__', *args)

    def __round__(self, *args):
        return self._math_unary('__round__', *args)

    def __trunc__(self, *args):
        return self._math_unary('__trunc__', *args)

    def __floor__(self, *args):
        return self._math_unary('__floor__', *args)

    def __ceil__(self, *args):
        return self._math_unary('__ceil__', *args)

    def __getattr__(self, attr):
        """
        Wrapper for arbitrary numeric methods.
        """
        # Raise error
        if attr == 'data' or self.data is None:
            return super().__getattribute__(attr)

        # Read from dataset
        data = self.data
        try:
            value = getattr(data, attr)
        except AttributeError:
            return super().__getattribute__(attr)  # raise error
        if callable(value):
            @functools.wraps(value)
            def func(*args, **kwargs):
                res = value(*args, **kwargs)
                if type(data) is type(res) or isinstance(res, numbers.Number):
                    new = copy.copy(self)
                    new.data = res
                    return new
                else:
                    return res
            return func
        else:
            return value

    def _math_other(self, attr, other, *args):  # power accepts modulo
        """
        Wrapper for standard mathematical operations.
        """
        if type(other) is type(self):
            other = other.data
        new = copy.copy(self)
        new.data = getattr(self.data, attr)(other, *args)
        return new

    def _math_inplace(self, attr, other, *args):  # power accepts modulo
        """
        Wrapper for inplace mathematical operations.
        """
        if type(other) is type(self):
            other = other.data
        self.data = getattr(self.data, attr)(other, *args)
        return self

    def _math_unary(self, attr, *args):  # round() accepts argument
        """
        Wrapper for unary operations.
        """
        new = copy.copy(self)
        new.data = getattr(self.data, attr)(*args)
        return new
