#! /usr/bin/env python

__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys

import numpy as np

import logging


logger = logging.getLogger(__name__)


def test_strang_splitting(plot_solution=False):
    from leap.rk import LSRK4MethodBuilder

    method1 = LSRK4MethodBuilder("y", rhs_func_name="<func>y1")
    method2 = LSRK4MethodBuilder("y", rhs_func_name="<func>y2")

    code1 = method1.generate()
    code2 = method2.generate()

    from leap.transform import strang_splitting

    code = strang_splitting(code1, code2, "primary")
    print(code)

    from utils import python_method_impl_codegen

    def exact(t):
        return np.exp(3j*t)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for dt in 2 ** -np.array(range(4, 7), dtype=np.float64):  # noqa pylint:disable=invalid-unary-operand-type
        interp = python_method_impl_codegen(code, function_map={
            "<func>y1": lambda t, y: 1j*y,
            "<func>y2": lambda t, y: 2j*y,
            })
        interp.set_up(t_start=0, dt_start=dt, context={"y": 1})

        final_t = 4
        times = []
        values = []
        for event in interp.run(t_end=final_t):
            if isinstance(event, interp.StateComputed):
                values.append(event.state_component)
                times.append(event.t)

        assert abs(times[-1] - final_t) / final_t < 0.1

        times = np.array(times)

        # Check that the timestep is preserved.
        assert np.allclose(np.diff(times), dt)

        if plot_solution:
            import matplotlib.pyplot as pt
            pt.plot(times, values, label="comp")
            pt.plot(times, exact(times), label="true")
            pt.show()

        error = abs(values[-1] - exact(final_t))
        eocrec.add_data_point(dt, error)

    print(eocrec.pretty_print())

    orderest = eocrec.estimate_order_of_convergence()[0, 1]
    assert orderest > 2 * 0.9


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
