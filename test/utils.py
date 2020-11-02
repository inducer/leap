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

import numpy as np
import logging

logger = logging.getLogger(__name__)


# {{{ things to pass for python_method_impl

def python_method_impl_interpreter(code, **kwargs):
    from dagrt.exec_numpy import NumpyInterpreter
    return NumpyInterpreter(code, **kwargs)


def python_method_impl_codegen(code, **kwargs):
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")
    return codegen.get_class(code)(**kwargs)

# }}}


def solver(f, t, sub_y, coeff, guess):
    from scipy.optimize import root
    return root(lambda unk: unk - f(t=t, y=sub_y + coeff*unk), guess).x


def solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                   pre_match={"unk": solve_var})
    pieces["guess"] = guess
    return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)


def execute_and_return_single_result(python_method_impl, code, initial_context={},
                                     max_steps=1):
    interpreter = python_method_impl(code, function_map={})
    interpreter.set_up(t_start=0, dt_start=0, context=initial_context)
    has_state_component = False
    for event in interpreter.run(max_steps=max_steps):
        if isinstance(event, interpreter.StateComputed):
            has_state_component = True
            state_component = event.state_component
    assert has_state_component
    return state_component


class Problem:
    """
    .. attribute :: t_start
    .. attribute :: t_end
    """

    def initial(self):
        """Return an initial value."""
        raise NotImplementedError()

    def exact(self, t):
        """Return the exact solution, if available."""
        raise NotImplementedError()

    def __call__(self, t, y):
        raise NotImplementedError()


class DefaultProblem(Problem):

    t_start = 1

    t_end = 10

    def initial(self):
        return np.array([1, 3], dtype=np.float64)

    def exact(self, t):
        inner = np.sqrt(3) / 2 * np.log(t)
        return np.sqrt(t) * (
                5 * np.sqrt(3) / 3 * np.sin(inner)
                + np.cos(inner)
                )

    def __call__(self, t, y):
        u, v = y
        return np.array([v, -u / t ** 2], dtype=np.float64)


_default_dts = 2 ** -np.array(range(4, 7), dtype=np.float64)  # noqa pylint:disable=invalid-unary-operand-type


def check_simple_convergence(method, method_impl, expected_order,
                             problem=DefaultProblem(), dts=_default_dts,
                             show_dag=False, plot_solution=False, implicit=False):
    component_id = method.component_id
    code = method.generate()
    #print(code)

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    if implicit:
        from leap.implicit import replace_AssignImplicit
        code = replace_AssignImplicit(code, {"solve": solver_hook})

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

        if implicit:
            from functools import partial
            interp = method_impl(code, function_map={
                "<func>" + component_id: problem,
                "<func>solver": partial(solver, problem),
                })
        else:
            interp = method_impl(code, function_map={
                "<func>" + component_id: problem,
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


def check_adaptive_timestep(python_method_impl, method, ss_frac, bs_frac,
                            show_dag=False, plot=False, implicit=False):
    # Use "DEBUG" to trace execution
    logging.basicConfig(level=logging.INFO)

    component_id = method.component_id
    code = method.generate()
    #print(code)
    #1/0

    if implicit:
        from leap.implicit import replace_AssignImplicit
        code = replace_AssignImplicit(code, {"solve": solver_hook})

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    from stiff_test_systems import VanDerPolProblem
    example = VanDerPolProblem()
    y = example.initial()

    if implicit:
        from functools import partial
        interp = python_method_impl(code,
                                    function_map={"<func>" + component_id: example,
                                    "<func>solver": partial(solver, example)})
    else:
        interp = python_method_impl(code,
                                    function_map={"<func>" + component_id: example})
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
        pt.clf()
        pt.plot(times, values[:, 1], "x-")
        pt.show()
        pt.plot(times, step_sizes, "x-")
        pt.show()

    step_sizes = np.array(step_sizes)
    small_step_frac = len(np.nonzero(step_sizes < 0.01)[0]) / len(step_sizes)
    big_step_frac = len(np.nonzero(step_sizes > 0.05)[0]) / len(step_sizes)

    print("small_step_frac (<0.01): %g - big_step_frac (>.05): %g"
            % (small_step_frac, big_step_frac))
    assert small_step_frac <= ss_frac, small_step_frac
    assert big_step_frac >= bs_frac, big_step_frac


# vim: foldmethod=marker
