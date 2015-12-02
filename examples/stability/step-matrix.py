from __future__ import division, print_function
import numpy as np
import numpy.linalg as la
from leap.rk import RK4Method  # noqa
from leap.multistep import AdamsBashforthMethod  # noqa


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    #method = RK4Method("y")
    method = AdamsBashforthMethod("y", order=3, static_dt=True)

    code = method.generate()

    print(code)

    def rhs_sym(t, y):
        return var("lambda")*y

    finder = StepMatrixFinder(code, function_map={"<func>y": rhs_sym},
            exclude_variables=["<p>step"])

    mat = finder.get_state_step_matrix("primary")

    print('Variables: %s' % finder.variables)
    np.set_printoptions(formatter={"all": str})
    print(mat)

    tol = 1e-8

    from fast_eval import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    def is_stable(direction, dt):
        smat = evaluate_mat({"<dt>": dt, "lambda": direction})

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
