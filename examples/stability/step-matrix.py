#!/usr/bin/env python
import numpy as np
import numpy.linalg as la
from leap.rk import RK4MethodBuilder  # noqa
from leap.multistep import AdamsBashforthMethodBuilder  # noqa


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    #method = RK4MethodBuilder("y")
    method = AdamsBashforthMethodBuilder("y", order=3, static_dt=True)

    code = method.generate()

    print(code)

    def rhs_sym(t, y):
        return var("lmbda")*y

    finder = StepMatrixFinder(code, function_map={"<func>y": rhs_sym},
            exclude_variables=["<p>step"])

    print(finder.get_maxima_expressions("primary"))
    mat = finder.get_phase_step_matrix("primary")

    print("Variables: %s" % finder.variables)
    np.set_printoptions(formatter={"all": str})
    print(mat)

    tol = 1e-8

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    def is_stable(direction, dt):
        smat = evaluate_mat({"<dt>": dt, "lmbda": direction})

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
