#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np
import numpy.linalg as la
from leap.multistep.multirate import TwoRateAdamsBashforthMethodBuilder
import matplotlib.pyplot as pt


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    speed_factor = 10
    method_name = "Fq"
    order = 3
    tol = 1e-8
    prec = 1e-5

    angles = np.linspace(0, 2*np.pi, 100, endpoint=False)

    for step_ratio in [1, 2, 3, 4, 5, 6]:
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

        from leap.step_matrix import fast_evaluator
        evaluate = fast_evaluator(mat)

        def is_stable(major_eigval, dt):
            smat = evaluate({
                        "<dt>": dt,
                        "f2f": major_eigval,
                        "s2f": 1/speed_factor,
                        "f2s": 1/speed_factor,
                        "s2s": major_eigval*1/speed_factor,
                        })

            eigvals = la.eigvals(smat)

            return (np.abs(eigvals) <= 1 + tol).all()

        from leap.stability import find_truth_bdry
        from functools import partial

        points = []

        for angle in angles:
            eigval = np.exp(1j*angle)

            max_dt = find_truth_bdry(partial(is_stable, eigval), prec=prec)

            stable_fake_eigval = eigval*max_dt

            points.append([stable_fake_eigval.real, stable_fake_eigval.imag])

        points = np.array(points).T

        pt.plot(points[0], points[1], "x", label="steprat: %d" % step_ratio)

    pt.legend(loc="best")
    pt.grid()

    outfile = "mr-stability-diagram.pdf"
    pt.savefig(outfile)
    print("Output written to %s" % outfile)


if __name__ == "__main__":
    main()
