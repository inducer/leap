from __future__ import division, print_function
import numpy as np
import numpy.linalg as la
from leap.multistep.multirate import TwoRateAdamsBashforthMethod


def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var
    from pymbolic import primitives as pm

    speed_factor = 12
    step_ratio = 2
    method_name = "F"
    order = 3

    print("speed factor: %g - step ratio: %g - method: %s "
            "- order: %d"
            % (speed_factor, step_ratio, method_name, order))

    method = TwoRateAdamsBashforthMethod(
            method=method_name, order=order, step_ratio=step_ratio,
            static_dt=True)

    code = method.generate()

    # Advection matrix sizes
    n_L = 175
    n_R = 50
    n = n_L + n_R
    
    # Add in Nek's global matrix from 1D advection/diffusion case
    M = np.loadtxt('system_matrix_advdif.txt',delimiter=',',ndmin=1)
    # Fast-to-fast block
    M_f2f_block = M[:n_L,:n_L]
    # Slow-to-slow block
    M_s2s_block = M[n_L:,n_L:]
    # Slow-to-fast block
    M_s2f_block = M[:n_L,n_L:]
    # Fast-to-slow block
    M_f2s_block = M[n_L:,:n_L]

    def f2f(t,f,s):
        print(f)
        1/0
        result = M_f2f_block.dot(f)
        return result
    finder = StepMatrixFinder(code,
            function_map={
                "<func>f2f": f2f,
                "<func>s2f": lambda t, f, s: M_s2f_block.dot(s),
                "<func>f2s": lambda t, f, s: M_f2s_block.dot(f),
                "<func>s2s": lambda t, f, s: M_s2s_block.dot(s),
                },
            exclude_variables=["<p>bootstrap_step"])
 
    print("Hi")
    print(code)
    shapes = {}
    mat = finder.get_state_step_matrix("primary", shapes=shapes)
    print("bye")

    print('Variables: %s' % finder.variables)
    np.set_printoptions(formatter={"all": str})
    print(mat)

    tol = 1e-8

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    def is_stable(direction, dt):
 
        smat = evaluate_mat({
                    "<dt>": dt,
                    "f2f": M_f2f_block,
                    "s2f": M_s2f_block,
                    "f2s": M_f2s_block,
                    "s2s": M_s2s_block,
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
