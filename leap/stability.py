"""Finding of stability regions."""

from __future__ import division, print_function

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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

import numpy as np
from cmath import pi


def is_stable(integrator_cls, k):
    def f(t, y):
        return k*y

    integrator = integrator_cls(function_map={"<func>y": f})
    integrator.set_up(t_start=0, dt_start=1, context={"y": 1})

    steps = 0
    last_state = None

    for event in integrator.run(t_end=None):
        if isinstance(event, integrator.StateComputed):
            steps += 1
            if steps > 100:
                return True
            last_state = event.state_component
            if abs(last_state) > 2:
                return False


def make_k_with_origin(origin, angle, mag):
    from cmath import exp
    return origin+mag*exp(1j*angle)


def refine(integrator_cls, make_k, angle, stable, unstable, prec):
    assert is_stable(integrator_cls, make_k(angle, stable))
    assert not is_stable(integrator_cls, make_k(angle, unstable))
    while abs(stable-unstable) > prec:
        mid = (stable+unstable)/2
        if is_stable(integrator_cls, make_k(angle, mid)):
            stable = mid
        else:
            unstable = mid
    else:
        return stable


def find_stability_bdry(code, prec, make_k, angle):
    # Generated code doesn't pickle well->generate here.
    from dagrt.codegen import PythonCodeGenerator
    integrator_cls = PythonCodeGenerator("Integrator").get_class(code)

    mag = 1

    if is_stable(integrator_cls, make_k(angle, mag)):
        # try to grow
        mag *= 2
        while is_stable(integrator_cls, make_k(angle, mag)):
            mag *= 2

            if mag > 2**8:
                return mag
        mag = refine(integrator_cls, make_k, angle, mag/2, mag, prec=prec)
    else:
        mag /= 2
        while not is_stable(integrator_cls, make_k(angle, mag)):
            mag /= 2

            if mag < prec:
                return mag
        mag = refine(integrator_cls, make_k, angle, mag, mag*2, prec=prec)

    return make_k(angle, mag)


def find_stability_region(code, parallel=None, n_angles=100, prec=1e-2,
        origin=-.3):
    if parallel is None:
        parallel = False

    angles = np.arange(0, 2*pi, 2*pi/n_angles)

    from functools import partial

    make_k = partial(make_k_with_origin, origin)
    find_stab = partial(find_stability_bdry, code, prec, make_k)

    if parallel:
        from multiprocessing import Pool
        points = Pool().map(find_stab, angles)
    else:
        points = list(map(find_stab, angles))

    return np.array(points)
