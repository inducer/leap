__copyright__ = """
Copyright (C) 2007-15 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

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
import numpy.linalg as la
import pytest
from pytools import memoize_method
from leap.multistep.multirate import (
        rhs_policy,
        MultiRateHistory as MRHistory,
        MultiRateMultiStepMethodBuilder,
        TwoRateAdamsBashforthMethodBuilder,
        TextualSchemeExplainer)


from utils import (  # noqa
        python_method_impl_interpreter as pmi_int,
        python_method_impl_codegen as pmi_cg)


class MultirateTimestepperAccuracyChecker:
    """Check that the multirate timestepper has the advertised accuracy."""

    def __init__(self, method, order, hist_length, step_ratio, static_dt, ode,
            method_impl, display_dag=False, display_solution=False):
        self.method = method
        self.order = order
        self.hist_length = hist_length
        self.step_ratio = step_ratio
        self.static_dt = static_dt
        self.ode = ode
        self.method_impl = method_impl
        self.display_dag = display_dag
        self.display_solution = display_solution

    @memoize_method
    def get_code(self, dt):
        method = TwoRateAdamsBashforthMethodBuilder(
                self.method, self.order, self.step_ratio,
                static_dt=self.static_dt,
                hist_consistency_threshold=1e-8,
                early_hist_consistency_threshold=dt**self.order,
                hist_length_slow=self.hist_length,
                hist_length_fast=self.hist_length)

        # Early consistency threshold checked for convergence
        # with timestep change - C. Mikida, 2/6/18 (commit hash 2e6ca077)

        # With method 4-Ss (limiting case), the following maximum relative
        # errors were observed:
        # for dt = 0.015625: 3.11E-09
        # for dt = 0.0078125: 1.04e-10
        # Corresponding EOC: 4.90

        # Reported relative errors show that no constant factor is needed on
        # early consistency threshold

        return method.generate()

    def initialize_method(self, dt):
        # Requires a coupled component.
        def make_coupled(f2f, f2s, s2f, s2s):
            def coupled(t, y):
                args = (t, y[0] + y[1], y[2] + y[3])
                return np.array((f2f(*args), f2s(*args), s2f(*args),
                    s2s(*args)),)
            return coupled

        function_map = {"<func>f2f": self.ode.f2f_rhs,
            "<func>s2f": self.ode.s2f_rhs, "<func>f2s": self.ode.f2s_rhs,
            "<func>s2s": self.ode.s2s_rhs, "<func>coupled": make_coupled(
                self.ode.f2f_rhs, self.ode.s2f_rhs, self.ode.f2s_rhs,
                self.ode.s2s_rhs)}

        print(self.get_code(dt))
        method = self.method_impl(self.get_code(dt), function_map=function_map)

        t = self.ode.t_start
        y = self.ode.initial_values
        method.set_up(t_start=t, dt_start=dt,
                context={"fast": y[0], "slow": y[1]})
        return method

    def get_error(self, dt, name=None, plot_solution=False):
        final_t = self.ode.t_end

        method = self.initialize_method(dt)

        times = []
        slow = []
        fast = []
        for event in method.run(t_end=final_t):
            if isinstance(event, method.StateComputed):
                if event.component_id == "slow":
                    slow.append(event.state_component)
                    times.append(event.t)
                elif event.component_id == "fast":
                    fast.append(event.state_component)

        assert abs(times[-1] - final_t) < 1e-10

        if 0:
            import matplotlib.pyplot as pt
            pt.plot(times, slow)
            pt.plot(times, self.ode.soln_1(times))
            pt.show()

        t = times[-1]
        y = (fast[-1], slow[-1])

        from multirate_test_systems import Basic, Tria

        if isinstance(self.ode, Basic) or isinstance(self.ode, Tria):
            # AK: why?
            if self.display_solution:
                self.plot_solution(times, fast, self.ode.soln_0)
            return abs(y[0]-self.ode.soln_0(t))
        else:
            from math import sqrt
            if self.display_solution:
                self.plot_solution(times, fast, self.ode.soln_0)
                self.plot_solution(times, slow, self.ode.soln_1)
            return abs(sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2))

    def show_dag(self, dt=2**(-6)):
        from dagrt.language import show_dependency_graph
        show_dependency_graph(self.get_code(dt))

    def plot_solution(self, times, values, soln, label=None):
        import matplotlib.pyplot as pt
        pt.plot(times, values, label="comp")
        pt.plot(times, soln(times), label="true")
        pt.legend(loc="best")
        pt.show()

    def __call__(self):
        """Run the test and output the estimated the order of convergence."""

        from pytools.convergence import EOCRecorder

        if self.display_dag:
            self.show_dag()

        eocrec = EOCRecorder()
        for n in range(6, 8):
            dt = 2**(-n)
            error = self.get_error(dt, "mrab-%d.dat" % self.order)
            eocrec.add_data_point(dt, error)

        print("------------------------------------------------------")
        print("ORDER %d" % self.order)
        print("------------------------------------------------------")
        print(eocrec.pretty_print())

        orderest = eocrec.estimate_order_of_convergence()[0, 1]
        assert orderest > self.order*0.70


@pytest.mark.slowtest
@pytest.mark.parametrize(("order", "hist_length"), [(1, 1),
    (3, 3), (3, 4), ])
@pytest.mark.parametrize("system", [
        #"Basic",
        "Full",
        #"Comp",
        #"Tria"
        ])
@pytest.mark.parametrize("method_name", TwoRateAdamsBashforthMethodBuilder.methods)
@pytest.mark.parametrize("static_dt", [True, False])
def test_multirate_accuracy(method_name, order, hist_length,
        system, static_dt, step_ratio=2):
    """Check that the multirate timestepper has the advertised accuracy"""

    import multirate_test_systems

    system = getattr(multirate_test_systems, system)

    print("------------------------------------------------------")
    print("METHOD: %s" % method_name)
    print("------------------------------------------------------")

    MultirateTimestepperAccuracyChecker(
        method_name, order, hist_length, step_ratio, static_dt=static_dt,
        ode=system(),
        method_impl=pmi_cg)()


def test_single_rate_identical(order=3, hist_length=3):
    from leap.multistep import AdamsBashforthMethodBuilder
    from dagrt.exec_numpy import NumpyInterpreter

    from multirate_test_systems import Full
    ode = Full()

    t_start = 0
    dt = 0.1

    # {{{ single rate

    single_rate_method = AdamsBashforthMethodBuilder("y", order=order,
            hist_length=hist_length)
    single_rate_code = single_rate_method.generate()

    def single_rate_rhs(t, y):
        f, s = y
        return np.array([
            ode.f2f_rhs(t, f, s)+ode.s2f_rhs(t, f, s),
            ode.f2s_rhs(t, f, s)+ode.s2s_rhs(t, f, s),
            ])

    single_rate_interp = NumpyInterpreter(
            single_rate_code,
            function_map={"<func>y": single_rate_rhs})

    single_rate_interp.set_up(t_start=t_start, dt_start=dt,
            context={"y": np.array([
                ode.soln_0(t_start),
                ode.soln_1(t_start),
                ])})

    single_rate_values = {}

    nsteps = 20

    for event in single_rate_interp.run():
        if isinstance(event, single_rate_interp.StateComputed):
            single_rate_values[event.t] = event.state_component

            if len(single_rate_values) == nsteps:
                break

    # }}}

    # {{{ two rate

    multi_rate_method = MultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(1, "<func>f", ("fast", "slow",),
                            hist_length=hist_length),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(1, "<func>s", ("fast", "slow",),
                            rhs_policy=rhs_policy.late, hist_length=hist_length),
                        ),),
                hist_consistency_threshold=1e-8,
                early_hist_consistency_threshold=dt**order)

    multi_rate_code = multi_rate_method.generate()

    def rhs_fast(t, fast, slow):
        return ode.f2f_rhs(t, fast, slow)+ode.s2f_rhs(t, fast, slow)

    def rhs_slow(t, fast, slow):
        return ode.f2s_rhs(t, fast, slow)+ode.s2s_rhs(t, fast, slow)

    multi_rate_interp = NumpyInterpreter(
            multi_rate_code,
            function_map={"<func>f": rhs_fast, "<func>s": rhs_slow})

    multi_rate_interp.set_up(t_start=t_start, dt_start=dt,
            context={
                "fast": ode.soln_0(t_start),
                "slow": ode.soln_1(t_start),
                })

    multi_rate_values = {}

    for event in multi_rate_interp.run():
        if isinstance(event, single_rate_interp.StateComputed):
            idx = {"fast": 0, "slow": 1}[event.component_id]
            if event.t not in multi_rate_values:
                multi_rate_values[event.t] = [None, None]

            multi_rate_values[event.t][idx] = event.state_component

            if len(multi_rate_values) > nsteps:
                break

    # }}}

    times = sorted(single_rate_values)
    single_rate_values = np.array([single_rate_values[t] for t in times])
    multi_rate_values = np.array([multi_rate_values[t] for t in times])
    print(single_rate_values)
    print(multi_rate_values)

    diff = la.norm((single_rate_values-multi_rate_values).reshape(-1))

    assert diff < 1e-13


@pytest.mark.parametrize("method_name", ["F", "Fqsr", "Srsf", "S"])
def test_2rab_scheme_explainers(method_name, order=3, step_ratio=3,
        explainer=TextualSchemeExplainer()):  # noqa: B008
    method = TwoRateAdamsBashforthMethodBuilder(
            method_name, order=order, step_ratio=step_ratio)

    method.generate(explainer=explainer)
    print(explainer)


def test_mrab_scheme_explainers(order=3, step_ratio=3,
        explainer=TextualSchemeExplainer()):  # noqa: B008
    method = MultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(1, "<func>f", ("fast", "slow",)),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(step_ratio, "<func>s", ("fast", "slow",),
                            rhs_policy=rhs_policy.late),
                        ),)
                )

    method.generate(explainer=explainer)
    print(explainer)


def test_mrab_with_derived_state_scheme_explainers(order=3, step_ratio=3,
        explainer=TextualSchemeExplainer()):  # noqa: B008
    method = MultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(1, "<func>f", ("fast", "slow",)),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(step_ratio, "<func>s", ("fast", "slow", "derived"),
                            rhs_policy=rhs_policy.late),
                        ),
                    (
                        "derived", "=",
                        MRHistory(step_ratio, "<func>compute_derived",
                            ("fast", "slow",), rhs_policy=rhs_policy.late),
                        ),
                    )
                )

    code = method.generate(explainer=explainer)
    print(code)
    print()
    print(explainer)


def test_dot(order=3, step_ratio=3, method_name="F", show=False):
    method = TwoRateAdamsBashforthMethodBuilder(
            method_name, order=order, step_ratio=step_ratio,
            hist_consistency_threshold=1e-8)
    code = method.generate()

    from dagrt.language import get_dot_dependency_graph
    print(get_dot_dependency_graph(code))

    if show:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)


@pytest.mark.parametrize(
    "fast_interval, slow_interval",
    (
        (1, 2),
        (3, 4)))
def test_two_rate_intervals(fast_interval, slow_interval, order=3):
    # Solve
    # f' = f+s
    # s' = -f+s

    def true_f(t):
        return np.exp(t)*np.sin(t)

    def true_s(t):
        return np.exp(t)*np.cos(t)

    method = MultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(fast_interval, "<func>f", ("fast", "slow",)),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(slow_interval, "<func>s", ("fast", "slow"))
                        ),
                    ),
                static_dt=True)

    code = method.generate()
    print(code)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")

    stepper_cls = codegen.get_class(code)

    for n in range(4, 7):
        t = 0
        dt = fast_interval * slow_interval * 2**(-n)
        final_t = 10

        stepper = stepper_cls(
                function_map={
                    "<func>f": lambda t, fast, slow: fast + slow,
                    "<func>s": lambda t, fast, slow: -fast + slow,
                    })

        stepper.set_up(
                t_start=t, dt_start=dt,
                context={
                    "fast": true_f(t),
                    "slow": true_s(t),
                    })

        f_times = []
        f_values = []
        s_times = []
        s_values = []
        for event in stepper.run(t_end=final_t):
            if isinstance(event, stepper_cls.StateComputed):
                if event.component_id == "fast":
                    f_times.append(event.t)
                    f_values.append(event.state_component)
                elif event.component_id == "slow":
                    s_times.append(event.t)
                    s_values.append(event.state_component)
                else:
                    raise ValueError(event.component_id)

        f_times = np.array(f_times)
        s_times = np.array(s_times)
        f_values_true = true_f(f_times)
        s_values_true = true_s(s_times)

        f_err = f_values - f_values_true
        s_err = s_values - s_values_true

        error = (
                la.norm(f_err) / la.norm(f_values_true)
                +  # noqa: W504
                la.norm(s_err) / la.norm(s_values_true))

        eocrec.add_data_point(dt, error)

    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > order * 0.7


def test_dependent_state(order=3, step_ratio=3):
    # Solve
    # f' = f+s
    # s' = -f+s

    def true_f(t):
        return np.exp(t)*np.sin(t)

    def true_s(t):
        return np.exp(t)*np.cos(t)

    method = MultiRateMultiStepMethodBuilder(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MRHistory(1, "<func>f", ("two_fast", "slow",)),
                        ),
                    (
                        "dt", "slow", "=",
                        MRHistory(step_ratio, "<func>s", ("fast", "slow"))
                        ),
                    (
                        "two_fast", "=",
                        MRHistory(step_ratio, "<func>twice", ("fast",)),
                        ),
                    ),
                static_dt=True)

    code = method.generate()
    print(code)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")

    stepper_cls = codegen.get_class(code)

    for n in range(4, 7):
        t = 0
        dt = 2**(-n)
        final_t = 10

        stepper = stepper_cls(
                function_map={
                    "<func>f": lambda t, two_fast, slow: 0.5*two_fast + slow,
                    "<func>s": lambda t, fast, slow: -fast + slow,
                    "<func>twice": lambda t, fast: 2*fast,
                    })

        stepper.set_up(
                t_start=t, dt_start=dt,
                context={
                    "fast": true_f(t),
                    "slow": true_s(t),
                    })

        f_times = []
        f_values = []
        s_times = []
        s_values = []
        for event in stepper.run(t_end=final_t):
            if isinstance(event, stepper_cls.StateComputed):
                if event.component_id == "fast":
                    f_times.append(event.t)
                    f_values.append(event.state_component)
                elif event.component_id == "slow":
                    s_times.append(event.t)
                    s_values.append(event.state_component)
                else:
                    raise ValueError(event.component_id)

        f_times = np.array(f_times)
        s_times = np.array(s_times)
        f_values_true = true_f(f_times)
        s_values_true = true_s(s_times)

        f_err = f_values - f_values_true
        s_err = s_values - s_values_true

        error = (
                la.norm(f_err) / la.norm(f_values_true)
                +  # noqa: W504
                la.norm(s_err) / la.norm(s_values_true))

        eocrec.add_data_point(dt, error)

    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 3*0.95


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
