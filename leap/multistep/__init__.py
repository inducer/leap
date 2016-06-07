"""Adams-Bashforth ODE solvers."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
Copyright (C) 2014, 2015 Matt Wala
Copyright (C) 2015 Cory Mikida
"""

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

import six.moves
import numpy as np
import numpy.linalg as la
from leap import Method
from pymbolic import var


__doc__ = """
.. autoclass:: AdamsBashforthMethod
"""


# {{{ Adams-Bashforth integration (with and without dynamic time steps)

def _linear_comb(coefficients, vectors):
    from operator import add
    return six.moves.reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))


class ABIntegrationFunctionFamily(object):
    def __len__(self):
        raise NotImplementedError()

    def evaluate(self, func_idx, x):
        raise NotImplementedError()

    def antiderivative(self, func_idx, x):
        raise NotImplementedError()


class ABMonomialIntegrationFunctionFamily(ABIntegrationFunctionFamily):
    def __init__(self, order):
        self.order = order

    def __len__(self):
        return self.order

    def evaluate(self, func_idx, x):
        return x**func_idx

    def antiderivative(self, func_idx, x):
        return 1/(func_idx+1) * x**(func_idx+1)


def emit_ab_integration(cb, name_gen,
        function_family, time_values, hist_vars, t_start, t_end):
    if isinstance(time_values, var):
        # {{{ variable time step
        hist_len = len(hist_vars)

        nfunctions = len(function_family)

        array = var("<builtin>array")
        linear_solve = var("<builtin>linear_solve")

        # use:
        # Vandermonde^T * ab_coeffs = integrate(t_start, t_end, monomials)

        vdmt = var(name_gen("vdm_transpose"))
        cb(vdmt, array(nfunctions*hist_len))

        coeff_rhs = var(name_gen("coeff_rhs"))
        cb(coeff_rhs, array(hist_len))

        j = var(name_gen("vdm_j"))

        for i in range(len(function_family)):
            cb(vdmt[i + j*nfunctions], function_family.evaluate(i, time_values[j]),
                loops=[(j.name, 0, hist_len)])

        for i in range(len(function_family)):
            cb(
                    coeff_rhs[i],
                    function_family.antiderivative(i, t_end)
                    - function_family.antiderivative(i, t_start))

        ab_coeffs = var(name_gen("ab_coeffs"))
        cb(ab_coeffs, linear_solve(vdmt, coeff_rhs, nfunctions, 1))

        return _linear_comb(
                    [ab_coeffs[ii] for ii in range(hist_len)],
                    hist_vars)

        # }}}

    else:
        # {{{ static time step

        hist_len = len(hist_vars)
        nfunctions = len(function_family)

        vdm_t = np.zeros((nfunctions, hist_len))
        coeff_rhs = np.zeros((nfunctions))

        for i in range(nfunctions):
            for j in range(hist_len):
                vdm_t[i, j] = function_family.evaluate(i, time_values[j])

            coeff_rhs[i] = (
                    function_family.antiderivative(i, t_end)
                    - function_family.antiderivative(i, t_start))

        ab_coeffs = la.solve(vdm_t, coeff_rhs)

        return _linear_comb(ab_coeffs, hist_vars)

        # }}}

# }}}


# {{{ ab method

class AdamsBashforthMethod(Method):
    """
    User-supplied context:
        <state> + component_id: The value that is integrated
        <func> + component_id: The right hand side

    .. automethod:: __init__
    .. automethod:: generate
    """

    def __init__(self, component_id, order, state_filter_name=None,
            hist_length=None, static_dt=False):
        """
        :arg static_dt: If *True*, changing the timestep during time integration
            is not allowed.
        """

        super(AdamsBashforthMethod, self).__init__()
        self.order = order

        if hist_length is None:
            hist_length = order

        self.hist_length = hist_length
        self.static_dt = static_dt

        self.component_id = component_id

        # Declare variables
        self.step = var('<p>step')
        self.function = var('<func>' + component_id)
        self.history = \
            [var('<p>f_n_minus_' + str(i)) for i in range(hist_length - 1, 0, -1)]

        if not self.static_dt:
            self.time_history = [
                    var('<p>t_n_minus_' + str(i))
                    for i in range(hist_length - 1, 0, -1)]

        self.state = var('<state>' + component_id)
        self.t = var('<t>')
        self.dt = var('<dt>')

        if state_filter_name is not None:
            self.state_filter = var("<func>" + state_filter_name)
        else:
            self.state_filter = None

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        from dagrt.language import DAGCode, CodeBuilder

        array = var("<builtin>array")
        rhs_var = var("rhs_var")

        # Initialization
        with CodeBuilder(label="initialization") as cb_init:
            cb_init(self.step, 1)

        # Primary
        with CodeBuilder(label="primary") as cb_primary:

            if not self.static_dt:
                time_history_data = self.time_history + [self.t]
                time_hist_var = var(name_gen("time_history"))
                cb_primary(time_hist_var, array(self.hist_length))
                for i in range(self.hist_length):
                    cb_primary(time_hist_var[i], time_history_data[i] - self.t)

                time_hist = time_hist_var
                t_end = self.dt
                dt_factor = 1

            else:
                time_hist = list(range(-self.hist_length+1, 0+1))
                dt_factor = self.dt
                t_end = 1

            cb_primary(rhs_var, self.eval_rhs(self.t, self.state))
            cb_primary.fence()
            history = self.history + [rhs_var]

            ab_sum = emit_ab_integration(
                            cb_primary, name_gen,
                            ABMonomialIntegrationFunctionFamily(self.order),
                            time_hist, history,
                            0, t_end)

            state_est = self.state + dt_factor * ab_sum
            if self.state_filter is not None:
                state_est = self.state_filter(state_est)
            cb_primary(self.state, state_est)

            # Rotate history and time history.
            for i in range(self.hist_length - 1):
                cb_primary.fence()
                cb_primary(self.history[i], history[i + 1])

                if not self.static_dt:
                    cb_primary(self.time_history[i], time_history_data[i + 1])

                cb_primary.fence()
            cb_primary(self.t, self.t + self.dt)
            cb_primary.yield_state(expression=self.state,
                                   component_id=self.component_id,
                                   time_id='', time=self.t)

        if self.hist_length == 1:
            # The first order method requires no bootstrapping.
            return DAGCode(
                states={
                    "initial": cb_init.as_execution_state(next_state="primary"),
                    "primary": cb_primary.as_execution_state(next_state="primary")
                    },
                initial_state="initial")

        # Bootstrap
        with CodeBuilder(label="bootstrap") as cb_bootstrap:
            self.rk_bootstrap(cb_bootstrap)
            cb_bootstrap(self.t, self.t + self.dt)
            cb_bootstrap.yield_state(expression=self.state,
                                     component_id=self.component_id,
                                     time_id='', time=self.t)
            cb_bootstrap(self.step, self.step + 1)
            with cb_bootstrap.if_(self.step, "==", self.hist_length):
                cb_bootstrap.state_transition("primary")

        return DAGCode(
                states={
                    "initialization": cb_init.as_execution_state("bootstrap"),
                    "bootstrap": cb_bootstrap.as_execution_state("bootstrap"),
                    "primary": cb_primary.as_execution_state("primary"),
                    },
                initial_state="initialization")

    def eval_rhs(self, t, y):
        """Return a node that evaluates the RHS at the given time and
        component value."""
        from pymbolic.primitives import CallWithKwargs
        return CallWithKwargs(function=self.function,
                              parameters=(),
                              kw_parameters={"t": t, self.component_id: y})

    def rk_bootstrap(self, cb):
        """Initialize the timestepper with an RK method."""

        rhs_var = var("rhs_var")

        cb(rhs_var, self.eval_rhs(self.t, self.state))

        # Save the current RHS to the AB history

        for i in range(len(self.history)):
            with cb.if_(self.step, "==", i + 1):
                cb(self.history[i], rhs_var)

                if not self.static_dt:
                    cb(self.time_history[i], self.t)

        from leap.rk import ORDER_TO_RK_METHOD
        rk_method = ORDER_TO_RK_METHOD[self.order]
        rk_tableau = tuple(zip(rk_method.c, rk_method.a_explicit))
        rk_coeffs = rk_method.output_coeffs

        # Stage loop (taken from EmbeddedButcherTableauMethod)
        rhss = [var("rk_rhs_" + str(i)) for i in range(len(rk_tableau))]
        for stage_num, (c, coeffs) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                cb(rhss[stage_num], rhs_var)
            else:
                stage = self.state + sum(self.dt * coeff * rhss[j]
                                         for (j, coeff)
                                         in enumerate(coeffs))

                if self.state_filter is not None:
                    stage = self.state_filter(stage)

                cb(rhss[stage_num], self.eval_rhs(self.t + c * self.dt, stage))

        # Merge the values of the RHSs.
        rk_comb = sum(coeff * rhss[j] for j, coeff in enumerate(rk_coeffs))
        cb.fence()

        state_est = self.state + self.dt * rk_comb
        if self.state_filter is not None:
            state_est = self.state_filter(state_est)

        # Assign the value of the new state.
        cb(self.state, state_est)

# }}}

# vim: fdm=marker
