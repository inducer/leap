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

from leap.multistep import AdaptiveBDFMethodBuilder
from stiff_test_systems import VanDerPolProblem, KapsProblem
import numpy as np

import logging

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)

logger = logging.getLogger(__name__)


def kaps_solver(f, t, sub_y, coeff, guess):
    from scipy.optimize import root
    return root(lambda unk: unk - f(t=t, y=sub_y + coeff*unk), guess).x


def VDPJac(t, y):
    jac = np.zeros((2, 2))
    mu = 30
    jac[0, 0] = 0
    jac[0, 1] = 1
    jac[1, 0] = -2*mu*y[0]*y[1] - 1
    jac[1, 1] = -mu*(y[0]*y[0] - 1)
    return jac


def newton_solver(t, sub_y, coeff, guess):

    vdp = VanDerPolProblem()
    d = 0
    corr_norm = 1.0
    reltol = 1e-6
    abstol = 0
    y_old = guess.copy()
    y_guess = guess.copy()
    corr_weights = np.zeros(2)
    # Match Scipy BDF
    newton_tol = 0.01
    newton_maxiter = 4
    corr_norm_old = None
    converged = False
    # Try pulling this out of the loop.
    jac = VDPJac(t, y_old)
    import scipy.linalg as la
    lu = la.lu_factor(np.eye(2) - coeff*jac, overwrite_a=True)
    # Check convergence w/weighted norm...
    for j in range(0, 2):
        corr_weights[j] = (reltol * np.abs(y_old[j]) + abstol)
    psi = -(sub_y - guess)
    for i in range(newton_maxiter):
        rhs = vdp(t, y_guess)
        corr = la.lu_solve(lu, coeff*rhs - psi - d, overwrite_b=True)
        y_guess += corr
        d += corr
        # RMS norm:
        corr_norm = np.linalg.norm(corr / corr_weights) / 2 ** 0.5
        if corr_norm_old is not None:
            rate = corr_norm / corr_norm_old
        else:
            rate = None
        if rate is not None and rate / (1 - rate) * corr_norm < newton_tol:
            converged = True
            break
        corr_norm_old = corr_norm

    if converged is False:
        raise ValueError("Newton failed to converge")

    # Return RHS that gives y_n+1 = y_guess
    return (y_guess + psi - y_old)/coeff


def solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                    pre_match={"unk": solve_var})
    pieces["guess"] = guess
    return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)


@pytest.mark.parametrize("problem, method, expected_order", [
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=1, max_dt_growth=10), 1],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=1, max_dt_growth=10, ndf=True), 1],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=2, max_dt_growth=10, bootstrap_factor=10), 2],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=2, max_dt_growth=10, bootstrap_factor=10, ndf=True), 2],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=3, max_dt_growth=10, bootstrap_factor=100), 3],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=3, max_dt_growth=10, bootstrap_factor=100, ndf=True), 3],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=4, max_dt_growth=10, bootstrap_factor=1000), 4],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=4, max_dt_growth=10, bootstrap_factor=1000, ndf=True), 4],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=5, max_dt_growth=10, bootstrap_factor=20000), 5],
    [KapsProblem(epsilon=0.9), AdaptiveBDFMethodBuilder(
        "y", fixed_order=5, max_dt_growth=10, bootstrap_factor=20000, ndf=True), 5],
    ])
def test_convergence(python_method_impl, problem, method, expected_order):
    pytest.importorskip("scipy")

    code = method.generate()

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in range(3, 8):
        dt = 2**(-n)

        y_0 = problem.initial()
        t_start = problem.t_start
        t_end = problem.t_end

        from functools import partial
        interp = python_method_impl(code, function_map={
            "<func>y": problem,
            "<func>solver": partial(kaps_solver, problem),
        })

        interp.set_up(t_start=t_start, dt_start=dt, context={"y": y_0})

        times = []
        values = []

        for event in interp.run(t_end=t_end):
            if isinstance(event, interp.StateComputed):
                values.append(event.state_component)
                times.append(event.t)

        times = np.array(times)
        values = np.array(values)

        assert abs(times[-1] - t_end) < 1e-10

        times = np.array(times)

        error = np.linalg.norm(values[-1] - problem.exact(t_end))
        eocrec.add_data_point(dt, error)

    print("------------------------------------------------------")
    print("expected order %d" % expected_order)
    print("------------------------------------------------------")
    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 0.9 * expected_order


# {{{ adaptive test

@pytest.mark.parametrize("method", [
    AdaptiveBDFMethodBuilder(
        "y", rtol=1e-6),
    AdaptiveBDFMethodBuilder(
        "y", rtol=1e-6, ndf=True),
    ])
def test_adaptive_timestep(python_method_impl, method, show_dag=False,
                           plot=False):
    pytest.importorskip("scipy")
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = method.component_id
    code = method.generate()

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    example = VanDerPolProblem()
    y = example.initial()

    interp = python_method_impl(code,
                                function_map={"<func>" + component_id: example,
                                              "<func>solver": newton_solver,
                                              })
    interp.set_up(t_start=example.t_start, dt_start=1e-5, context={component_id: y})

    times = []
    values = []

    new_times = []
    new_values = []

    last_t = 0
    step_sizes = []

    for event in interp.run(t_end=example.t_end):
        if isinstance(event, interp.StateComputed):
            assert event.component_id == component_id

            new_values.append(event.state_component)
            new_times.append(event.t)
        elif isinstance(event, interp.StepCompleted):
            if not new_times:
                continue

            step_sizes.append(event.t - last_t)
            last_t = event.t

            times.extend(new_times)
            values.extend(new_values)
            del new_times[:]
            del new_values[:]
        elif isinstance(event, interp.StepFailed):
            del new_times[:]
            del new_values[:]

            logger.info("failed step at t=%s" % event.t)

    times = np.array(times)
    values = np.array(values)
    step_sizes = np.array(step_sizes)

    if plot:
        import matplotlib.pyplot as pt
        pt.plot(times, values[:, 1], "x-")
        pt.title("Van Der Pol: State 2")
        pt.show()
        pt.plot(times, step_sizes, "x-")
        pt.title("Van Der Pol: Step Sizes")
        pt.show()

    step_sizes = np.array(step_sizes)
    small_step_frac = len(np.nonzero(step_sizes < 0.01)[0]) / len(step_sizes)
    big_step_frac = len(np.nonzero(step_sizes > 0.05)[0]) / len(step_sizes)

    print("small_step_frac (<0.01): %g - big_step_frac (>.05): %g"
            % (small_step_frac, big_step_frac))
    assert small_step_frac <= 0.7, small_step_frac
    assert big_step_frac >= 0.2, big_step_frac

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
