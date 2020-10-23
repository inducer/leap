#!/usr/bin/env python
"""This example illustrates the use of a custom solver."""
import numpy as np


class KapsProblem:
    """
    From Kennedy and Carpenter, Section 7.1

    y_1' = - (epsilon^{-1} + 2) y_1 + epsilon^{-1} y_2^2
    y_2' = y_1 - y_2 - y_2^2

    0 <= t <= 1

    The initial conditions are

      y_1 = y_2 = 1.

    The exact solution is

      y_1 = exp(-2t)
      y_2 = exp(-t).

    The stiff component are the terms multiplied by epsilon^{-1}.
    """

    def __init__(self, epsilon):
        self._epsilon_inv = 1 / epsilon
        self.t_start = 0
        self.t_end = 1

    def initial(self):
        return np.array([1., 1.])

    def nonstiff(self, t, y):
        y_1 = y[0]
        y_2 = y[1]
        return np.array([-2 * y_1, y_1 - y_2 - y_2 ** 2])

    def stiff(self, t, y):
        y_1 = y[0]
        y_2 = y[1]
        return np.array([-self._epsilon_inv * (y_1 - y_2 ** 2), 0])

    def jacobian(self, t, y):
        y_2 = y[1]
        return np.array([
            [-self._epsilon_inv - 2, 2 * self._epsilon_inv * y_2],
            [1, -1 - 2 * y_2]
        ])

    def exact(self, t):
        return np.array([np.exp(-2 * t), np.exp(-t)])


_atol = 1.0e-5


def solver(f, j, t, u_n, x, c):
    """Kennedy and Carpenter, page 15"""
    import numpy.linalg as nla
    Id = np.eye(len(u_n))       # noqa: N806
    u = u_n
    while True:
        M = Id - c * j(t, u)    # noqa N806
        r = -(u - u_n) + x + c * f(t=t, y=u)
        d = nla.solve(M, r)
        u = u + d
        if 0.005 * _atol >= nla.norm(d):
            return f(t=t, y=u)


def solver_hook(solve_expr, solve_var, solver_id, guess):
    from dagrt.expression import match, substitute

    pieces = match("unk - <func>rhs(t=t, y=<state>y + sub_y + coeff*unk)",
                   solve_expr,
                   bound_variable_names=["<state>y"],
                   pre_match={"unk": solve_var})

    pieces["guess"] = guess

    return substitute("<func>solver(t, <state>y, sub_y, coeff)", pieces)


def run():
    from leap.rk.imex import KennedyCarpenterIMEXARK4MethodBuilder
    from dagrt.codegen import PythonCodeGenerator

    # Construct the method generator.
    mgen = KennedyCarpenterIMEXARK4MethodBuilder("y", atol=_atol)

    # Generate the code for the method.
    code = mgen.generate()

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})
    cls = PythonCodeGenerator("IMEXIntegrator").get_class(code)

    # Set up the problem and run the method.
    from functools import partial
    problem = KapsProblem(epsilon=0.001)
    integrator = cls(function_map={
        "<func>expl_y": problem.nonstiff,
        "<func>impl_y": problem.stiff,
        "<func>solver": partial(solver, problem.stiff, problem.jacobian),
        "<func>j": problem.jacobian})

    integrator.set_up(t_start=problem.t_start,
                      dt_start=1.0e-1,
                      context={"y": problem.initial()})

    t = None
    y = None

    for event in integrator.run(t_end=problem.t_end):
        if isinstance(event, integrator.StateComputed):
            t = event.t
            y = event.state_component

    print("Error: " + str(np.linalg.norm(y - problem.exact(t))))


if __name__ == "__main__":
    run()
