#! /usr/bin/env python

__copyright__ = "Copyright (C) 2014 Matt Wala"

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


import numpy as np
import pytest
import sys

from leap.multistep import AdaptiveSBDFMethodBuilder
from stiff_test_systems import KapsProblem

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


def solver(f, t, sub_y, coeff, guess, expl_rhs):
    from scipy.optimize import root
    return root(lambda unk: unk - expl_rhs - f(t=t, y=sub_y + coeff*unk), guess).x


def solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - (expl_rhs + <func>rhs(t=t, y=sub_y + coeff*unk))",
                   solve_expr, pre_match={"unk": solve_var})
    pieces["guess"] = guess
    return substitute("<func>solver(t, sub_y, coeff, guess, expl_rhs)", pieces)


@pytest.mark.parametrize("problem, method, expected_order", [
    [KapsProblem(epsilon=0.9), AdaptiveSBDFMethodBuilder(
        "y", fixed_order=1, max_dt_growth=10), 1],
    [KapsProblem(epsilon=0.9), AdaptiveSBDFMethodBuilder(
        "y", fixed_order=2, max_dt_growth=10, bootstrap_factor=1), 2],
    [KapsProblem(epsilon=0.9), AdaptiveSBDFMethodBuilder(
        "y", fixed_order=3, max_dt_growth=10, bootstrap_factor=100), 3],
    [KapsProblem(epsilon=0.9), AdaptiveSBDFMethodBuilder(
        "y", fixed_order=4, max_dt_growth=10, bootstrap_factor=1000), 4],
    [KapsProblem(epsilon=0.9), AdaptiveSBDFMethodBuilder(
        "y", fixed_order=5, max_dt_growth=10, bootstrap_factor=10000), 5],
    ])
def test_convergence(python_method_impl, problem, method, expected_order,
                     show_dag=False, plot_solution=False):
    pytest.importorskip("scipy")
    from functools import partial
    dts = 2**-np.array(range(3, 8), dtype=np.float64)
    function_map = {"<func>expl_y": problem.nonstiff,
                    "<func>impl_y": problem.stiff,
                    "<func>solver": partial(solver, problem.stiff)}
    from utils import check_simple_convergence
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=expected_order, show_dag=show_dag,
                             plot_solution=plot_solution,
                             function_map=function_map, dts=dts,
                             implicit=True, problem=problem,
                             solver_hook=solver_hook)


@pytest.mark.parametrize("problem, method, rhs_extrap", [
    [KapsProblem(epsilon=0.001), AdaptiveSBDFMethodBuilder, True],
    [KapsProblem(epsilon=0.001), AdaptiveSBDFMethodBuilder, False],
    ])
def test_adaptive(python_method_impl, problem, method, rhs_extrap):
    pytest.importorskip("scipy")

    t_start = problem.t_start
    t_end = problem.t_end
    dt = 1.0e-1

    tols = [10.0 ** (-j) for j in range(1, 5)]

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    # Test that tightening the tolerance will decrease the overall error.
    for atol in tols:
        generator = method("y", atol=atol, rhs_extrap=rhs_extrap)
        code = generator.generate()

        from leap.implicit import replace_AssignImplicit
        code = replace_AssignImplicit(code, {"solve": solver_hook})

        from functools import partial
        interp = python_method_impl(code, function_map={
            "<func>expl_y": problem.nonstiff,
            "<func>impl_y": problem.stiff,
            "<func>solver": partial(solver, problem.stiff)
        })
        interp.set_up(t_start=t_start, dt_start=dt,
                      context={"y": problem.initial()})

        times = []
        values = []

        new_times = []
        new_values = []

        for event in interp.run(t_end=t_end):
            clear_flag = False
            if isinstance(event, interp.StateComputed):
                assert event.component_id == "y"
                new_values.append(event.state_component)
                new_times.append(event.t)
            elif isinstance(event, interp.StepCompleted):
                values.extend(new_values)
                times.extend(new_times)
                clear_flag = True
            elif isinstance(event, interp.StepFailed):
                clear_flag = True
            if clear_flag:
                del new_times[:]
                del new_values[:]

        times = np.array(times)
        values = np.array(values)
        exact = problem.exact(times[-1])
        error = np.linalg.norm(values[-1] - exact)
        eocrec.add_data_point(atol, error)

    print("Error vs. tolerance")
    print(eocrec.pretty_print())
    order = eocrec.estimate_order_of_convergence()[0, 1]
    assert order > 0.9


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
