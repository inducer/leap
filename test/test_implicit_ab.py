#! /usr/bin/env python
from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2020 Cory Mikida"

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


import numpy as np
import pytest
import sys

from leap.multistep import AdamsMoultonMethodBuilder

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


def am_solver(f, t, sub_y, coeff, guess):
    from scipy.optimize import root
    return root(lambda unk: unk - f(t=t, y=sub_y + coeff*unk), guess).x


def am_solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                   pre_match={"unk": solve_var})
    pieces["guess"] = guess
    return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)


@pytest.mark.parametrize(("method", "expected_order", "static_dt"), [
    (AdamsMoultonMethodBuilder("y", order, static_dt=static_dt), order, static_dt)
    for order in [1, 2, 3, 4, 5]
    for static_dt in [True, False]
    ] + [
    (AdamsMoultonMethodBuilder("y", order, hist_length=order+1,
        static_dt=static_dt), order, static_dt)
    for order in [1, 2, 3, 4, 5]
    for static_dt in [True, False]
    ])
def test_am_convergence(python_method_impl, method, expected_order, static_dt):
    plot_solution = False
    pytest.importorskip("scipy")
    from utils import DefaultProblem
    problem = DefaultProblem()

    component_id = method.component_id
    code = method.generate()

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": am_solver_hook})

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    # default dts
    dts = 2 ** -np.array(range(4, 7), dtype=np.float64)  # noqa pylint:disable=invalid-unary-operand-type

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

        from functools import partial
        interp = python_method_impl(code, function_map={
            "<func>" + component_id: problem,
            "<func>solver": partial(am_solver, problem),
            })
        interp.set_up(t_start=t, dt_start=dt, context={component_id: y})

        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, interp.StateComputed):
                assert event.component_id == component_id
                values.append(event.state_component[0])
                times.append(event.t)

        assert abs(times[-1] - final_t) / final_t < 0.1

        times = np.array(times)

        if plot_solution:
            import matplotlib.pyplot as pt
            pt.plot(times, values, label="comp")
            pt.plot(times, problem.exact(times), label="true")
            pt.show()

        error = abs(values[-1] - problem.exact(final_t))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("%s: expected order %d" % (method.__class__.__name__,
                                     expected_order))
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > expected_order * 0.9


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
