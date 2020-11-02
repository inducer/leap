#! /usr/bin/env python

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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

from leap.rk import (
        ODE23MethodBuilder, ODE45MethodBuilder,
        ForwardEulerMethodBuilder,
        MidpointMethodBuilder, HeunsMethodBuilder,
        RK3MethodBuilder, RK4MethodBuilder, RK5MethodBuilder,
        LSRK4MethodBuilder,
        SSPRK22MethodBuilder, SSPRK33MethodBuilder,
        )
from leap.rk.imex import KennedyCarpenterIMEXARK4MethodBuilder

import logging

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)

logger = logging.getLogger(__name__)


# {{{ non-adaptive test

# test using
# python test_rk.py 'test_rk_accuracy(pmi_int, ODE23MethodBuilder("y", use_high_order=False), 2)'  # noqa

@pytest.mark.parametrize(("method", "expected_order"), [
    (ODE23MethodBuilder("y", use_high_order=False), 2),
    (ODE23MethodBuilder("y", use_high_order=True), 3),
    (ODE45MethodBuilder("y", use_high_order=False), 4),
    (ODE45MethodBuilder("y", use_high_order=True), 5),
    (ForwardEulerMethodBuilder("y"), 1),
    (MidpointMethodBuilder("y"), 2),
    (HeunsMethodBuilder("y"), 2),
    (RK3MethodBuilder("y"), 3),
    (RK4MethodBuilder("y"), 4),
    (RK5MethodBuilder("y"), 5),
    (LSRK4MethodBuilder("y"), 4),
    (KennedyCarpenterIMEXARK4MethodBuilder("y", use_implicit=False,
        explicit_rhs_name="y"), 4),
    (SSPRK22MethodBuilder("y"), 2),
    (SSPRK33MethodBuilder("y"), 3),
    ])
def test_rk_accuracy(python_method_impl, method, expected_order,
                     show_dag=False, plot_solution=False):
    from utils import check_simple_convergence
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=expected_order, show_dag=show_dag,
                             plot_solution=plot_solution)

# }}}


# {{{ adaptive test

@pytest.mark.parametrize("method", [
    ODE23MethodBuilder("y", rtol=1e-6),
    ODE45MethodBuilder("y", rtol=1e-6),
    KennedyCarpenterIMEXARK4MethodBuilder("y", rtol=1e-6, use_implicit=False,
        explicit_rhs_name="y"),
    ])
def test_adaptive_timestep(python_method_impl, method, show_dag=False,
                           plot=False):
    from utils import check_adaptive_timestep
    check_adaptive_timestep(python_method_impl=python_method_impl, method=method,
                             ss_frac=0.35, bs_frac=0.16, show_dag=show_dag,
                             plot=plot, implicit=False)

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
