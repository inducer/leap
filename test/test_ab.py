#! /usr/bin/env python


__copyright__ = "Copyright (C) 2014 Andreas Kloeckner, Matt Wala"

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

# avoid spurious: pytest.mark.parametrize is not callable
# pylint: disable=not-callable

import sys
import pytest
from leap.multistep import AdamsBashforthMethodBuilder

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


@pytest.mark.parametrize(("method", "expected_order"), [
    (AdamsBashforthMethodBuilder("y", order, static_dt=static_dt), order)
    for order in [1, 3, 5]
    for static_dt in [True, False]
    ] + [
    (AdamsBashforthMethodBuilder("y", order, hist_length=order+1,
        static_dt=static_dt), order)
    for order in [1, 3, 5]
    for static_dt in [True, False]
    ])
def test_ab_accuracy(python_method_impl, method, expected_order,
        show_dag=False, plot_solution=False):
    from utils import check_simple_convergence
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=expected_order, show_dag=show_dag,
                             plot_solution=plot_solution)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
