"""Various useful functions for working with the timestepper description
language."""

from __future__ import division, with_statement

__copyright__ = """Copyright (C) 2014 Matt Wala"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from pymbolic.mapper.dependency import DependencyMapper


class VectorAwareDependencyMapper(DependencyMapper):

    def map_vector(self, expr):
        return self.combine(map(self.rec, expr.children))

    def map_foreign(self, expr):
        if expr is None or isinstance(expr, str):
            return frozenset()
        else:
            return super(VectorAwareDependencyMapper, self).map_foreign(expr)


variable_mapper = VectorAwareDependencyMapper(composite_leaves=False)


def get_variables(expr):
    """Returns the set of names of variables used in the expression."""
    return frozenset(dep.name for dep in variable_mapper(expr))


def is_state_variable(var):
    """Check if the given name corresponds to a state variable."""
    if var == '<t>' or var == '<dt>':
        return True
    elif var.startswith('<state>') or var.startswith('<p>'):
        return True
    else:
        return False


def peek(s):
    """Return a single element from a non-empty iterable."""
    return s.__iter__().next()


def get_unique_name(prefix, *name_sets):
    """Return a name that begins with prefix and is not found in any of the
    name sets."""

    def is_in_name_sets(var):
        for name_set in name_sets:
            if var in name_set:
                return True
        return False

    prefix = str(prefix)
    if not is_in_name_sets(prefix):
        return prefix

    suffix = 0
    while is_in_name_sets(prefix + str(suffix)):
        suffix += 1
    return prefix + str(suffix)