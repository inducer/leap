from __future__ import division, print_function
import numpy as np
import numpy.linalg as la
from leap.multistep.multirate import TwoRateAdamsBashforthMethod
import sys

def main():
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var
    from pymbolic import primitives as pm
    from pymbolic.interop import maxima
    from pymbolic.interop.maxima import MaximaKernel, MaximaParser
    from pymbolic.interop.maxima import MaximaStringifyMapper 
    
    speed_factor = 12
    step_ratio = int(sys.argv[1])
    method_name = "Fq"
    order = 3

    print("speed factor: %g - step ratio: %g - method: %s "
            "- order: %d"
            % (speed_factor, step_ratio, method_name, order))

    method = TwoRateAdamsBashforthMethod(
            method=method_name, order=order, step_ratio=step_ratio,
            static_dt=True)

    code = method.generate()

    # Advection matrix sizes
    n_L = 72
    n_R = 6
    n = n_L + n_R
    
    # Add in Nek's global matrix from 1D advection/diffusion case
    M = np.loadtxt('system_matrix_adv.txt',delimiter=',',ndmin=1)
    # Fast-to-fast block
    M_f2f_block = M[:n_L,:n_L]
    # Slow-to-slow block
    M_s2s_block = M[n_L:,n_L:]
    # Slow-to-fast block
    M_s2f_block = M[:n_L,n_L:]
    # Fast-to-slow block
    M_f2s_block = M[n_L:,:n_L]

    finder = StepMatrixFinder(code,
            function_map={
                "<func>f2f": lambda f, s, t: M_f2f_block.dot(f),
                "<func>s2f": lambda f, s, t: M_s2f_block.dot(s),
                "<func>f2s": lambda f, s, t: M_f2s_block.dot(f),
                "<func>s2s": lambda f, s, t: M_s2s_block.dot(s),
                },
            exclude_variables=["<p>bootstrap_step"])
 
    shapes = {'<p>hist_fast_rhs0_hist_0_ago': n_L,
            '<p>hist_fast_rhs0_hist_1_ago': n_L,
            '<p>hist_fast_rhs0_hist_2_ago': n_L, 
            '<p>hist_fast_rhs1_hist_0_ago': n_L, 
            '<p>hist_fast_rhs1_hist_1_ago': n_L, 
            '<p>hist_fast_rhs1_hist_2_ago': n_L, 
            '<p>hist_slow_rhs0_hist_0_ago': n_R, 
            '<p>hist_slow_rhs0_hist_1_ago': n_R, 
            '<p>hist_slow_rhs0_hist_2_ago': n_R, 
            '<p>hist_slow_rhs1_hist_0_ago': n_R, 
            '<p>hist_slow_rhs1_hist_1_ago': n_R, 
            '<p>hist_slow_rhs1_hist_2_ago': n_R, 
            '<state>fast': n_L, 
            '<state>slow': n_R}

    #mat = finder.get_state_step_matrix("primary", shapes=shapes)
    #print(finder.get_maxima_expressions("primary", shapes=shapes))

    # Feed this to Maxima to get the symbolic step matrix
    maxima.set_debug(1)
    max_kernel = MaximaKernel()
    msm = MaximaStringifyMapper()
    finder.get_maxima_expressions("primary", shapes=shapes)
    max_kernel._initialize

    max_kernel.eval_str('batch("maxima_in.txt")$')

    # Get the resulting step matrix back from Maxima
    maxima_mat = max_kernel.eval_str("jacobian(after_step,initial)")
    
    # Convert the result to a numpy array
    mat = np.fromstring(maxima_mat)

    tol = 1e-8

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(mat)

    def is_stable(direction, dt):

        smat = evaluate_mat({
                    "<dt>": dt,
                    "f2f": M_f2f_block*direction,
                    "s2f": M_s2f_block,
                    "f2s": M_f2s_block,
                    "s2s": M_s2s_block*direction,
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
