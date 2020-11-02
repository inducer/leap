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

from leap.multistep import EmbeddedAdamsMethodBuilder
import numpy as np

import logging

from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)

logger = logging.getLogger(__name__)


# {{{ non-adaptive test

@pytest.mark.parametrize(("method", "expected_order"), [
    (EmbeddedAdamsMethodBuilder("y", order=2, use_high_order=False), 2),
    (EmbeddedAdamsMethodBuilder("y", order=3, use_high_order=False), 3),
    (EmbeddedAdamsMethodBuilder("y", order=4, use_high_order=False), 4),
    ])
def test_embedded_accuracy(python_method_impl, method, expected_order,
                     show_dag=False, plot_solution=False):
    from utils import check_simple_convergence
    check_simple_convergence(method=method, method_impl=python_method_impl,
                             expected_order=expected_order, show_dag=show_dag,
                             plot_solution=plot_solution, implicit=True)

# }}}

# {{{ adaptive test


def solver(f, t, sub_y, coeff, guess):
    from scipy.optimize import root
    return root(lambda unk: unk - f(t=t, y=sub_y + coeff*unk), guess).x


def solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                   pre_match={"unk": solve_var})
    pieces["guess"] = guess
    return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)


@pytest.mark.parametrize(("method", "ss_frac", "bs_frac"), [
    (EmbeddedAdamsMethodBuilder("y", order=2, rtol=1e-6), 0.5, 0.05),
    (EmbeddedAdamsMethodBuilder("y", order=3, rtol=1e-6), 0.5, 0.01),
    (EmbeddedAdamsMethodBuilder("y", order=4, rtol=1e-6), 0.8, 0.0005),
    ])
def test_adaptive_timestep(python_method_impl, method, ss_frac, bs_frac,
                           show_dag=False, plot=False):
    from utils import check_adaptive_timestep
    check_adaptive_timestep(python_method_impl=python_method_impl, method=method,
                             ss_frac=ss_frac, bs_frac=bs_frac, show_dag=show_dag,
                             plot=plot, implicit=True)


@pytest.mark.parametrize(("method", "expected_order"),  [
    (EmbeddedAdamsMethodBuilder("y", order=2, rtol=1e-6), 2),
    (EmbeddedAdamsMethodBuilder("y", order=3, rtol=1e-6), 3),
    (EmbeddedAdamsMethodBuilder("y", order=4, rtol=1e-6), 4),
    ])
def test_adaptive_accuracy(method, expected_order, show_dag=False,
                           plot=False, python_method_impl=pmi_cg):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = method.component_id
    code = method.generate()

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})

    code_nonadapt = EmbeddedAdamsMethodBuilder("y", order=expected_order).generate()
    code_nonadapt = replace_AssignImplicit(code_nonadapt, {"solve": solver_hook})

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    from stiff_test_systems import VanDerPolProblem
    example = VanDerPolProblem()
    y = example.initial()

    from functools import partial
    interp = python_method_impl(code,
                                function_map={"<func>" + component_id: example,
                                "<func>solver": partial(solver, example)})

    interp_nonadapt = python_method_impl(code_nonadapt,
                                function_map={"<func>" + component_id: example,
                                "<func>solver": partial(solver, example)})

    interp.set_up(t_start=example.t_start, dt_start=1e-5, context={component_id: y})

    times = []
    values = []

    new_times = []
    new_values = []

    last_t = 0
    step_sizes = []
    nsteps = []
    istep = 0

    # Initial run to establish step sizes.
    for event in interp.run(t_end=example.t_end/10.0):
        if isinstance(event, interp.StateComputed):
            assert event.component_id == component_id

            new_values.append(event.state_component)
            new_times.append(event.t)
        elif isinstance(event, interp.StepCompleted):
            if not new_times:
                continue

            istep += 1
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
    nsteps = len(step_sizes)
    final_time = times[-1]
    final_val = values[-1]
    end_vals = []
    end_vals.append(final_val)
    dts = []
    dts.append(10.0/nsteps)
    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()
    for i in range(1, 5):
        fac = 2**i
        nsteps_run = fac*nsteps
        dt = np.zeros(nsteps_run)

        for j in range(0, nsteps_run, fac):
            dt[j] = step_sizes[int(j/fac)]/fac
            for k in range(1, fac):
                dt[j+k] = step_sizes[int(j/fac)]/fac

        # Now that we have our new set of timesteps, do the run,
        # same as before, but with adaptivity turned off.
        interp_nonadapt.set_up(t_start=example.t_start, dt_start=dt[0],
                               context={component_id: y})
        iout = 1
        for event in interp_nonadapt.run(t_end=final_time):
            if isinstance(event, interp_nonadapt.StateComputed):
                assert event.component_id == component_id

                end_val = event.state_component
            elif isinstance(event, interp_nonadapt.StepCompleted):
                if iout < nsteps*fac:
                    if event.t + dt[iout] >= final_time:
                        interp_nonadapt.dt = final_time - event.t
                    else:
                        interp_nonadapt.dt = dt[iout]
                    iout += 1
                else:
                    interp_nonadapt.dt = final_time - event.t

        end_vals.append(end_val)
        dts.append(10.0/iout)

    # Now calculate errors using the final time as the
    # true solution (self-convergence)
    for i in range(1, 5):
        eocrec.add_data_point(dts[i-1],
                              np.linalg.norm(end_vals[i-1] - end_vals[-1]))

    print(eocrec.pretty_print())
    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 0.9 * expected_order


# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
