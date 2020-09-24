#!/usr/bin/env python
import numpy as np
import numpy.linalg as la
from leap.multistep.multirate import TwoRateAdamsBashforthMethodBuilder


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    speed_factor = 10
    step_ratio = 7
    method_name = "Fq"
    order = 3

    print("speed factor: %g - step ratio: %g - method: %s "
            "- order: %d"
            % (speed_factor, step_ratio, method_name, order))

    method = TwoRateAdamsBashforthMethodBuilder(
            method=method_name, order=order, step_ratio=step_ratio,
            static_dt=True)

    code = method.generate()

    finder = StepMatrixFinder(code,
            function_map={
                "<func>f2f": lambda t, f, s: var("f2f") * f,
                "<func>s2f": lambda t, f, s: var("s2f") * s,
                "<func>f2s": lambda t, f, s: var("f2s") * f,
                "<func>s2s": lambda t, f, s: var("s2s") * s,
                },
            exclude_variables=["<p>bootstrap_step"])

    mat = finder.get_phase_step_matrix("primary")

    if 0:
        print("Variables: %s" % finder.variables)
        np.set_printoptions(formatter={"all": str})
        print(mat)

    tol = 1e-8

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    def is_stable(direction, dt):
        smat = evaluate_mat({
                    "<dt>": dt,
                    "f2f": direction,
                    "s2f": 1/speed_factor,
                    "f2s": 1/speed_factor,
                    "s2s": direction*1/speed_factor,
                    })

        eigvals = la.eigvals(smat)

        return (np.abs(eigvals) <= 1 + tol).all()

    from leap.stability import find_truth_bdry
    from functools import partial

    prec = 1e-5
    print("stable imaginary timestep:",
            find_truth_bdry(partial(is_stable, 1j), prec=prec))
    print("stable neg real timestep:",
            find_truth_bdry(partial(is_stable, -1), prec=prec))


if __name__ == "__main__":
    main()
