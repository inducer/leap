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


# {{{ things to pass for python_method_impl

def python_method_impl_interpreter(code, **kwargs):
    from dagrt.exec_numpy import NumpyInterpreter
    return NumpyInterpreter(code, **kwargs)


def python_method_impl_codegen(code, **kwargs):
    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")
    return codegen.get_class(code)(**kwargs)

# }}}


def execute_and_return_single_result(python_method_impl, code, initial_context=None,
                                     max_steps=1):
    if initial_context is None:
        initial_context = {}
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
                             problem=None, dts=_default_dts,
                             show_dag=False, plot_solution=False):
    if problem is None:
        problem = DefaultProblem()

    component_id = method.component_id
    code = method.generate()
    print(code)

    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

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


# vim: foldmethod=marker
