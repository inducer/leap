#!/usr/bin/env python
"""This example demonstrates direct construction of an adaptive RK method using
an order 2/3 method pair.

Source:

    P. Bogacki and L. F. Shampine.
    A 3(2) pair of Runge-Kutta formulas.
    Appl. Math. Lett., 2(4):321–325, 1989.
    doi: 10.1016/0893-9659(89)90079-7.

The same functionality is supported using leap.rk.ODE23MethodBuilder.

"""

import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class KapsProblem:
    """
    Source: Section 7.1 of

        Christopher A. Kennedy and Mark H. Carpenter.
        Additive Runge-Kutta schemes for convection-diffusion-reaction
            equations.
        Appl. Numer. Math., 44 (1-2):139–181, 2003.
        doi: 10.1016/S0168-9274(02)00138-1.

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
            [1, -1 - 2 * y_2]])

    def exact(self, t):
        return np.array([np.exp(-2 * t), np.exp(-t)])

    def __call__(self, t, y):
        return self.stiff(t, y) + self.nonstiff(t, y)


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


def demo_rk_adaptive():
    from dagrt.language import CodeBuilder
    from pymbolic import var

    # Set tolerance
    tol = 1e-3
    # Bulk declare symbolic names
    k1, k2, k3, k4, t, dt, dt_old, y, y_hi, y_lo, f, norm = (
        var(name) for name in
        "k1 k2 k3 k4 <t> <dt> dt_old <state>y y_hi y_lo <func>f <builtin>norm_inf".split())  # noqa

    with CodeBuilder("primary") as cb:
        # Calculate the RK stage values
        cb(k1, f(t, y))
        cb(k2, f(t + 1/2 * dt, y + dt * (1/2 * k1)))
        cb(k3, f(t + 3/4 * dt, y + dt * (3/4 * k2)))
        cb(k4, f(t + dt, y + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)))
        # Compute the low and high order solutions
        cb(y_lo, y + dt * (7/24 * k1 + 1/4 * k2 + 1/3 * k3 + 1/8 * k4))
        cb(y_hi, y + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3))
        # Save the value of dt
        cb(dt_old, dt)
        # Update dt based on the error estimate
        err_est = norm(y_lo - y_hi)
        order = 3
        cb(dt, 0.9 * dt * (tol / err_est) ** (1 / order))
        # Adapt the step size
        with cb.if_(err_est, "<=", tol):
            # Update t and y
            cb(y, y_hi)
            cb(t, t + dt_old)
            cb.yield_state(expression=y, component_id="y", time=t, time_id=None)  # noqa
        with cb.else_():
            cb.fail_step()

    from dagrt.language import DAGCode
    code = DAGCode(phases={
        "primary": cb.as_execution_phase(next_phase="primary")
        },
        initial_phase="primary")

    print(code)

    # Generate and run the method.
    from dagrt.codegen import PythonCodeGenerator
    cls = PythonCodeGenerator("RKAdaptiveMethod").get_class(code)
    eocrec = get_convergence_data(cls, problem=KapsProblem(0.001))
    print(eocrec.pretty_print())


if __name__ == "__main__":
    demo_rk_adaptive()
