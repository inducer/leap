#!/usr/bin/env python
"""This example demonstrates direct construction of a semi-implicit RK method.

Source:

    Roger Alexander.
    Diagonally implicit Runge–Kutta methods for stiff O.D.E.'s.
    SIAM Journal on Numerical Analysis, 14(6):1006--1021, 1977.
    doi: 10.1137/0714068.

"""


import numpy as np
import logging


logging.basicConfig(level=logging.INFO)


class SimpleDecayProblem:

    t_start = 0
    t_end = 0.5

    def initial(self):
        return np.array([1])

    def exact(self, t):
        return np.array([np.exp(-10 * t)])

    def __call__(self, t, y):
        return -10 * y


_default_dts = 2 ** -np.array(range(5, 10), dtype=np.float64)


def get_convergence_data(method_class, problem, dts=_default_dts):
    from pytools.convergence import EOCRecorder

    eocrec = EOCRecorder()

    component_id = "y"

    for dt in dts:
        t = problem.t_start
        y = problem.initial()
        final_t = problem.t_end

        interp = method_class(function_map={
                "<func>f": problem,
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

        error = abs(values[-1] - problem.exact(final_t)[0])
        eocrec.add_data_point(dt, error)

    return eocrec


def demo_rk_implicit():
    from dagrt.language import CodeBuilder
    from pymbolic import var

    k1, k2, t, dt, y, f = (
        var(name) for name in
        "k1 k2 <t> <dt> <state>y <func>f".split())

    gamma = (2 - 2**0.5) / 2

    with CodeBuilder("primary") as cb:
        cb.assign_implicit_1(
            k1,
            solve_component=k1,
            expression=k1 - f(t + gamma * dt, y + dt * gamma * k1),
            guess=f(t, y))
        cb.assign_implicit_1(
            k2,
            solve_component=k2,
            expression=k2 - f(t + dt, y + dt * ((1 - gamma) * k1 + gamma * k2)),  # noqa
            guess=k1)
        cb(y, y + dt * ((1 - gamma) * k1 + gamma * k2))
        cb(t, t + dt)
        cb.yield_state(y, "y", t, None)

    from dagrt.language import DAGCode
    code = DAGCode(phases={
        "primary": cb.as_execution_phase(next_phase="primary")
        },
        initial_phase="primary")

    def solver_hook(solve_expr, unknown, solver_id, guess):
        from dagrt.expression import match, substitute
        pieces = match(
            "k - <func>rhs(time, y + dt * (c0 + c1 * k))",
            solve_expr,
            pre_match={"k": unknown})
        return substitute("-10 * (dt * c0 + y) / (10 * dt * c1 + 1)", pieces)

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, solver_hook)
    print(code)

    from dagrt.codegen import PythonCodeGenerator
    cls = PythonCodeGenerator("IRKMethodBuilder").get_class(code)
    eocrec = get_convergence_data(cls, SimpleDecayProblem())
    print(eocrec.pretty_print())


if __name__ == "__main__":
    demo_rk_implicit()
