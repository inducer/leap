#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np
import numpy.linalg as la
from leap.multistep.multirate import TwoRateAdamsBashforthMethodBuilder
import matplotlib.pyplot as pt
from functools import partial


def process_eigval(evaluate_mat, speed_factor, prec, major_eigval):
    major_mag = abs(major_eigval)
    smat = evaluate_mat({
                "<dt>": 1,
                "f2f": major_eigval,
                "s2f": -1/speed_factor*major_mag,
                "f2s": -1/speed_factor*major_mag,
                "s2s": major_eigval*1/speed_factor,
                })

    eigvals = la.eigvals(smat)

    return np.max(np.abs(eigvals))


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    speed_factor = 10
    method_name = "Fq"
    order = 3
    step_ratio = 3
    prec = 1e-5

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

    left = -3
    right = 1
    bottom = -4
    top = 4
    res = 200

    points = np.mgrid[left:right:res*1j, bottom:top:res*1j]
    eigvals = points[0] + 1j*points[1]
    eigvals_flat = eigvals.reshape(-1)

    from multiprocessing import Pool
    pool = Pool()

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    stable_dts_list = pool.map(
            partial(process_eigval, evaluate_mat, speed_factor, prec), eigvals_flat)

    max_eigvals = np.zeros(eigvals.shape)
    max_eigvals.reshape(-1)[:] = stable_dts_list

    pt.title("speed factor: %g - step ratio: %g - method: %s "
            "- order: %d"
            % (speed_factor, step_ratio, method_name, order))

    log_max_eigvals = np.log10(1e-15+max_eigvals.T)
    pt.imshow(log_max_eigvals, extent=(left, right, bottom, top), cmap="viridis")
    pt.colorbar()
    pt.contour(eigvals.real, eigvals.imag, log_max_eigvals.T, [0], zorder=10)
    pt.gca().set_aspect("equal")
    pt.grid()

    outfile = "mr-max-eigvals.pdf"
    pt.savefig(outfile)

    print("Output written to %s" % outfile)


if __name__ == "__main__":
    main()
