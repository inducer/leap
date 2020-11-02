"""Adams-Bashforth ODE solvers."""


__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
Copyright (C) 2014, 2015 Matt Wala
Copyright (C) 2015, 2020 Cory Mikida
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

import numpy as np
import numpy.linalg as la
from leap import MethodBuilder, OneOrderAdaptiveMethodBuilderMixin
from pymbolic import var


__doc__ = """
.. autoclass:: AdamsIntegrationFunctionFamily
.. autoclass:: AdamsMonomialIntegrationFunctionFamily
.. autoclass:: AdamsMethodBuilder
.. autoclass:: AdamsBashforthMethodBuilder
.. autoclass:: AdamsMoultonMethodBuilder
"""


# {{{ Adams-Bashforth integration (with and without dynamic time steps)

def _linear_comb(coefficients, vectors):
    from operator import add
    from functools import reduce
    return reduce(add,
            (coeff * v for coeff, v in zip(coefficients, vectors)))


class AdamsIntegrationFunctionFamily:
    """An abstract interface for function families used for
    Adams-type time integration.

    .. automethod:: __len__
    .. automethod:: evaluate
    .. automethod:: antiderivative
    """

    def __len__(self):
        raise NotImplementedError()

    def evaluate(self, func_idx, x):
        raise NotImplementedError()

    def antiderivative(self, func_idx, x):
        raise NotImplementedError()


class AdamsMonomialIntegrationFunctionFamily(AdamsIntegrationFunctionFamily):
    """
    Implements :class:`AdamsMonomialIntegrationFunctionFamily`.
    """
    def __init__(self, order):
        self.order = order

    def __len__(self):
        return self.order

    def evaluate(self, func_idx, x):
        return x**func_idx

    def antiderivative(self, func_idx, x):
        return 1/(func_idx+1) * x**(func_idx+1)


def _emit_func_family_operation(cb, name_gen,
        function_family, time_values, hist_vars, rhs_func):
    if isinstance(time_values, var):
        # {{{ variable time step
        hist_len = len(hist_vars)

        nfunctions = len(function_family)

        array = var("<builtin>array")
        linear_solve = var("<builtin>linear_solve")
        svd = var("<builtin>svd")
        matmul = var("<builtin>matmul")
        transpose = var("<builtin>transpose")

        # use:
        # Vandermonde^T * a_coeffs = integrate(t_start, t_end, monomials)

        vdmt = var(name_gen("vdm_transpose"))
        cb(vdmt, array(nfunctions*hist_len))

        coeff_rhs = var(name_gen("coeff_rhs"))
        cb(coeff_rhs, array(nfunctions))

        j = var(name_gen("vdm_j"))

        for i in range(len(function_family)):
            cb(vdmt[i + j*nfunctions], function_family.evaluate(i, time_values[j]),
                loops=[(j.name, 0, hist_len)])

        for i in range(len(function_family)):
            cb(coeff_rhs[i], rhs_func(i))

        a_coeffs = var(name_gen("a_coeffs"))

        if hist_len == nfunctions:
            cb(a_coeffs, linear_solve(vdmt, coeff_rhs, nfunctions, 1))
        else:
            # Least squares with SVD builtin
            u = var(name_gen("u"))
            ut = var(name_gen("ut"))
            intermed = var(name_gen("intermed"))
            ainv = var(name_gen("ainv"))
            sigma = var(name_gen("sigma"))
            sig_array = var(name_gen("sig_array"))
            v = var(name_gen("v"))
            vt = var(name_gen("vt"))

            cb(ainv, array(nfunctions*hist_len))
            cb(intermed, array(nfunctions*hist_len))

            cb((u, sigma, vt), svd(vdmt, hist_len))

            cb(ut, transpose(u, nfunctions))
            cb(v, transpose(vt, hist_len))

            # Make singular value array
            cb(sig_array, array(nfunctions*nfunctions))

            for j in range(len(function_family)*len(function_family)):
                cb(sig_array[j], 0)

            for i in range(len(function_family)):
                cb(sig_array[i*(nfunctions+1)], sigma[i]**-1)

            cb(intermed, matmul(v, sig_array, nfunctions, nfunctions))
            cb(ainv, matmul(intermed, ut, nfunctions, nfunctions))
            cb(a_coeffs, matmul(ainv, coeff_rhs, nfunctions, 1))

        return _linear_comb(
                    [a_coeffs[ii] for ii in range(hist_len)],
                    hist_vars)

        # }}}

    else:
        # {{{ static time step

        hist_len = len(hist_vars)
        nfunctions = len(function_family)

        vdm_t = np.zeros((nfunctions, hist_len))
        coeff_rhs = np.zeros(nfunctions)

        for i in range(nfunctions):
            for j in range(hist_len):
                vdm_t[i, j] = function_family.evaluate(i, time_values[j])

            coeff_rhs[i] = rhs_func(i)

        if hist_len == nfunctions:
            a_coeffs = la.solve(vdm_t, coeff_rhs)
        else:
            # SVD-based least squares solve
            u, sigma, v = la.svd(vdm_t, full_matrices=False)
            ainv = np.dot(v.transpose(), np.dot(la.inv(np.diag(sigma)),
                u.transpose()))
            a_coeffs = np.dot(ainv, coeff_rhs)

        return _linear_comb(a_coeffs, hist_vars)

        # }}}


def emit_adams_integration(cb, name_gen,
        function_family, time_values, hist_vars, t_start, t_end):
    return _emit_func_family_operation(
            cb, name_gen, function_family, time_values, hist_vars,
            lambda i: (
                function_family.antiderivative(i, t_end)
                - function_family.antiderivative(i, t_start)))


def emit_adams_extrapolation(cb, name_gen,
        function_family, time_values, hist_vars, t_eval):
    return _emit_func_family_operation(
            cb, name_gen, function_family, time_values, hist_vars,
            lambda i: function_family.evaluate(i, t_eval))

# }}}


# {{{ adams method


class AdamsMethodBuilder(MethodBuilder):
    """
    User-supplied context:
        <state> + component_id: The value that is integrated
        <func> + component_id: The right hand side

    .. automethod:: __init__
    .. automethod:: generate
    """

    def __init__(self, component_id, function_family=None, state_filter_name=None,
            hist_length=None, static_dt=False, order=None, _extra_bootstrap=False):
        """
        :arg function_family: Accepts an instance of
            :class:`AdamsIntegrationFunctionFamily`
            or an integer, in which case the classical monomial function family
            with the order given by the integer is used.
        :arg static_dt: If *True*, changing the timestep during time integration
            is not allowed.
        """

        if function_family is not None and order is not None:
            raise ValueError("may not specify both function_family and order")

        if function_family is None:
            function_family = order
            del order

        if isinstance(function_family, int):
            function_family = AdamsMonomialIntegrationFunctionFamily(function_family)

        super().__init__()
        self.function_family = function_family

        if hist_length is None:
            hist_length = len(function_family)

        self.hist_length = hist_length
        self.static_dt = static_dt
        self.extra_bootstrap = _extra_bootstrap

        self.component_id = component_id

        # Declare variables
        self.step = var("<p>step")
        self.function = var("<func>" + component_id)
        self.history = \
            [var("<p>f_n_minus_" + str(i)) for i in range(hist_length - 1, 0, -1)]

        if not self.static_dt:
            self.time_history = [
                    var("<p>t_n_minus_" + str(i))
                    for i in range(hist_length - 1, 0, -1)]

        self.state = var("<state>" + component_id)
        self.t = var("<t>")
        self.dt = var("<dt>")

        self.state_filter_name = state_filter_name
        if state_filter_name is not None:
            self.state_filter = var("<func>" + state_filter_name)
        else:
            self.state_filter = None

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """

        from dagrt.language import DAGCode, CodeBuilder

        # Initialization
        with CodeBuilder(name="initialization") as cb_init:
            cb_init(self.step, 1)

        # Primary
        with CodeBuilder(name="primary") as cb_primary:
            self.generate_primary(cb_primary)

        if self.hist_length == 1:
            # The first order method requires no bootstrapping.
            return DAGCode(
                phases={
                    "initial": cb_init.as_execution_phase(next_phase="primary"),
                    "primary": cb_primary.as_execution_phase(next_phase="primary")
                    },
                initial_phase="initial")

        # Bootstrap
        with CodeBuilder(name="bootstrap") as cb_bootstrap:
            self.rk_bootstrap(cb_bootstrap)
            cb_bootstrap(self.t, self.t + self.dt)
            cb_bootstrap.yield_state(expression=self.state,
                                     component_id=self.component_id,
                                     time_id="", time=self.t)
            cb_bootstrap(self.step, self.step + 1)
            bootstrap_length = self.determine_bootstrap_length()
            with cb_bootstrap.if_(self.step, "==", bootstrap_length):
                cb_bootstrap.switch_phase("primary")

        return DAGCode(
                phases={
                    "initialization": cb_init.as_execution_phase("bootstrap"),
                    "bootstrap": cb_bootstrap.as_execution_phase("bootstrap"),
                    "primary": cb_primary.as_execution_phase("primary"),
                    },
                initial_phase="initialization")

    def eval_rhs(self, t, y):
        """Return a node that evaluates the RHS at the given time and
        component value."""
        from pymbolic.primitives import CallWithKwargs
        return CallWithKwargs(function=self.function,
                              parameters=(),
                              kw_parameters={"t": t, self.component_id: y})

    def rotate_and_yield(self, cb, hist, time_hist):
        for i in range(self.hist_length - 1):
            cb(self.history[i], hist[i + 1])

            if not self.static_dt:
                cb(self.time_history[i], time_hist[i + 1])

        cb(self.t, self.t + self.dt)
        cb.yield_state(expression=self.state,
                               component_id=self.component_id,
                               time_id="", time=self.t)

    def set_up_time_history(self, cb, new_t):
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()
        array = var("<builtin>array")
        if not self.static_dt:
            time_history_data = self.time_history + [new_t]
            time_hist_var = var(name_gen("time_history"))
            cb(time_hist_var, array(self.hist_length))
            for i in range(self.hist_length):
                cb(time_hist_var[i], time_history_data[i] - self.t)

            time_hist = time_hist_var
            t_end = self.dt
            dt_factor = 1

        else:
            if new_t == self.t:
                time_hist = list(range(-self.hist_length+1, 0+1))  # noqa pylint:disable=invalid-unary-operand-type
                time_history_data = list(range(-self.hist_length+1, 0+1))  # noqa pylint:disable=invalid-unary-operand-type
            else:
                time_hist = list(range(-self.hist_length+2, 0+2))  # noqa pylint:disable=invalid-unary-operand-type
                time_history_data = list(range(-self.hist_length+2, 0+2))  # noqa pylint:disable=invalid-unary-operand-type
            dt_factor = self.dt
            t_end = 1

        return time_history_data, time_hist, dt_factor, t_end

    def generate_primary(self, cb):
        raise NotImplementedError()

    def rk_bootstrap(self, cb):
        raise NotImplementedError()

    def determine_bootstrap_length(self):
        raise NotImplementedError()

# }}}


# {{{ ab method

class AdamsBashforthMethodBuilder(AdamsMethodBuilder):
    def generate_primary(self, cb):
        rhs_var = var("rhs_var")
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        time_history_data, time_hist, \
                dt_factor, t_end = self.set_up_time_history(cb, self.t)

        cb(rhs_var, self.eval_rhs(self.t, self.state))
        history = self.history + [rhs_var]

        ab_sum = emit_adams_integration(
                        cb, name_gen,
                        self.function_family,
                        time_hist, history,
                        0, t_end)

        state_est = self.state + dt_factor * ab_sum
        if self.state_filter is not None:
            state_est = self.state_filter(state_est)
        cb(self.state, state_est)

        # Rotate history and time history.
        self.rotate_and_yield(cb, history, time_history_data)

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

        from leap.rk import ORDER_TO_RK_METHOD_BUILDER
        rk_method = ORDER_TO_RK_METHOD_BUILDER[self.function_family.order]
        rk_coeffs = rk_method.output_coeffs
        stage_coeff_set_names = ("explicit",)
        stage_coeff_sets = {"explicit": rk_method.a_explicit}
        estimate_coeff_set_names = ("main",)
        estimate_coeff_sets = {"main": rk_coeffs}
        rhs_funcs = {"explicit": var("<func>"+self.component_id)}

        # Traverse RK stage loop of appropriate order and update state.
        rk = rk_method(self.component_id, self.state_filter_name)
        cb = rk.generate_butcher_init(cb, stage_coeff_set_names,
                                      stage_coeff_sets, rhs_funcs,
                                      estimate_coeff_set_names,
                                      estimate_coeff_sets)
        cb, rhss, est_vars = rk.generate_butcher_primary(cb, stage_coeff_set_names,
                                                         stage_coeff_sets, rhs_funcs,
                                                         estimate_coeff_set_names,
                                                         estimate_coeff_sets)

        # Assign the value of the new state.
        cb(self.state, est_vars[0])

    def determine_bootstrap_length(self):

        # In the explicit case, this is always
        # equal to history length.
        bootstrap_length = self.hist_length

        return bootstrap_length
# }}}


# {{{ am method


class AdamsMoultonMethodBuilder(AdamsMethodBuilder):
    def generate_primary(self, cb):

        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()
        rhs_next_var = var("rhs_next_var")
        rhs_var_to_unknown = {}
        unkvar = cb.fresh_var("unk")
        rhs_var_to_unknown[rhs_next_var] = unkvar

        # In implicit mode, the time history must
        # include the *next* point in time.
        time_history_data, time_hist, \
                dt_factor, t_end = self.set_up_time_history(cb, self.t + self.dt)

        # Implicit setup - rhs_next_var is an unknown, needs implicit solve.
        equations = []
        unknowns = set()
        knowns = set()

        unknowns.add(rhs_next_var)

        # Update history
        history = self.history + [rhs_next_var]

        # Set up the actual Adams-Moulton step.
        am_sum = emit_adams_integration(
                        cb, name_gen,
                        self.function_family,
                        time_hist, history,
                        0, t_end)

        state_est = self.state + dt_factor * am_sum

        # Build the implicit solve expression.
        from dagrt.expression import collapse_constants
        from pymbolic.mapper.distributor import DistributeMapper as DistMap
        solve_expression = collapse_constants(
                rhs_next_var - self.eval_rhs(self.t + self.dt,
                                                 DistMap()(state_est)),
                list(unknowns) + [self.state],
                cb.assign, cb.fresh_var)
        equations.append(solve_expression)

        # {{{ emit solve if possible

        if unknowns and len(unknowns) == len(equations):
            from leap.implicit import generate_solve
            generate_solve(cb, unknowns, equations, rhs_var_to_unknown, self.state)

        del equations[:]
        knowns.update(unknowns)
        unknowns.clear()

        # }}}

        # Update the state now that we've solved.
        if self.state_filter is not None:
            state_est = self.state_filter(state_est)
        cb(self.state, state_est)

        # Rotate history and time history.
        self.rotate_and_yield(cb, history, time_history_data)

    def rk_bootstrap(self, cb):
        """Initialize the timestepper with an IMPLICIT RK method."""

        from leap.rk import IMPLICIT_ORDER_TO_RK_METHOD_BUILDER
        rk_method = IMPLICIT_ORDER_TO_RK_METHOD_BUILDER[self.function_family.order]
        rk_coeffs = rk_method.output_coeffs
        stage_coeff_set_names = ("implicit",)
        stage_coeff_sets = {"implicit": rk_method.a_implicit}
        estimate_coeff_set_names = ("main",)
        estimate_coeff_sets = {"main": rk_coeffs}
        rhs_funcs = {"implicit": var("<func>"+self.component_id)}

        if self.extra_bootstrap:
            first_save_step = 2
        else:
            first_save_step = 1

        with cb.if_(self.step, "==", first_save_step):
            # Save the first RHS to the AM history
            rhs_var = var("rhs_var")

            cb(rhs_var, self.eval_rhs(self.t, self.state))
            cb(self.history[0], rhs_var)

            if not self.static_dt:
                cb(self.time_history[0], self.t)

        # Traverse RK stage loop of appropriate order and update state.
        rk = rk_method(self.component_id, self.state_filter_name)
        cb = rk.generate_butcher_init(cb, stage_coeff_set_names,
                                      stage_coeff_sets, rhs_funcs,
                                      estimate_coeff_set_names,
                                      estimate_coeff_sets)
        cb, rhss, est_vars = rk.generate_butcher_primary(cb, stage_coeff_set_names,
                                                         stage_coeff_sets, rhs_funcs,
                                                         estimate_coeff_set_names,
                                                         estimate_coeff_sets)

        # Assign the value of the new state.
        cb(self.state, est_vars[0])

        # Save the "next" RHS to the AM history
        rhs_next_var = var("rhs_next_var")

        cb(rhs_next_var, self.eval_rhs(self.t + self.dt, self.state))

        for i in range(1, len(self.history)):
            if self.extra_bootstrap:
                save_crit = i+1
            else:
                save_crit = i

            with cb.if_(self.step, "==", save_crit):
                cb(self.history[i], rhs_next_var)

                if not self.static_dt:
                    cb(self.time_history[i], self.t + self.dt)

    def determine_bootstrap_length(self):

        # In the implicit case, this is
        # equal to history length - 1, unless
        # we want an extra bootstrap step for
        # comparison with explicit methods.
        if self.extra_bootstrap:
            bootstrap_length = self.hist_length
        else:
            bootstrap_length = self.hist_length - 1

        return bootstrap_length

# }}}


# {{{ embedded method w/adaptivity


class EmbeddedAdamsMethodBuilder(
        AdamsMethodBuilder, OneOrderAdaptiveMethodBuilderMixin):
    """
    User-supplied context:
      <state> + component_id: The value that is integrated
      <func> + component_id: The right hand side function
    """

    def __init__(self, component_id, function_family=None, state_filter_name=None,
            hist_length=None, static_dt=False, order=None, _extra_bootstrap=False,
            use_high_order=False, atol=0, rtol=0, max_dt_growth=None,
            min_dt_shrinkage=None):
        """
        :arg function_family: Accepts an instance of
            :class:`AdamsIntegrationFunctionFamily`
            or an integer, in which case the classical monomial function family
            with the order given by the integer is used.
        :arg static_dt: If *True*, changing the timestep during time integration
            is not allowed.
        """

        if function_family is not None and order is not None:
            raise ValueError("may not specify both function_family and order")

        if function_family is None:
            function_family = order
            del order

        if isinstance(function_family, int):
            function_family = AdamsMonomialIntegrationFunctionFamily(function_family)

        self.function_family = function_family

        if hist_length is None:
            hist_length = len(function_family) + 1

        # Check for reasonable history length.
        if hist_length < len(function_family) + 1:
            raise ValueError("Invalid history length specified for embedded Adams")

        self.hist_length = hist_length

        # If adaptivity is on, we can't have a static timestep.
        if atol or rtol:
            if static_dt is True:
                raise ValueError("Can't have static timestepping with adaptivity")

        self.static_dt = static_dt
        self.extra_bootstrap = _extra_bootstrap

        self.component_id = component_id

        # Declare variables
        self.step = var("<p>step")
        self.function = var("<func>" + component_id)
        self.history = \
            [var("<p>f_n_minus_" + str(i)) for i in range(hist_length - 1, 0, -1)]

        if not self.static_dt:
            self.time_history = [
                    var("<p>t_n_minus_" + str(i))
                    for i in range(hist_length - 1, 0, -1)]

        self.state = var("<state>" + component_id)
        self.t = var("<t>")
        self.dt = var("<dt>")

        self.state_filter_name = state_filter_name
        if state_filter_name is not None:
            self.state_filter = var("<func>" + state_filter_name)
        else:
            self.state_filter = None

        OneOrderAdaptiveMethodBuilderMixin.__init__(
                self,
                atol=atol,
                rtol=rtol,
                max_dt_growth=max_dt_growth,
                min_dt_shrinkage=min_dt_shrinkage)

        self.use_high_order = use_high_order

    def generate_primary(self, cb):

        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()
        array = var("<builtin>array")
        rhs_next_var = var("rhs_next_var")
        rhs_var_to_unknown = {}
        unkvar = cb.fresh_var("unk")
        rhs_var_to_unknown[rhs_next_var] = unkvar

        # In implicit mode, the time history must
        # include the *next* point in time.
        time_history_data, time_hist, \
                dt_factor, t_end = self.set_up_time_history(cb, self.t + self.dt)

        # Implicit setup - rhs_next_var is an unknown, needs implicit solve.
        equations = []
        unknowns = set()
        knowns = set()

        unknowns.add(rhs_next_var)

        # Update history
        history = self.history + [rhs_next_var]

        # Create history to feed to AB.
        history_ab = history[:-1]
        time_hist_ab_var = var(name_gen("time_history_ab"))
        cb(time_hist_ab_var, array(self.hist_length-1))
        for i in range(self.hist_length-1):
            cb(time_hist_ab_var[i], time_hist[i])

        time_hist_ab = time_hist_ab_var

        # Create history to feed to AM.
        history_am = history[1:]
        time_hist_am_var = var(name_gen("time_history_am"))
        cb(time_hist_am_var, array(self.hist_length-1))
        for i in range(self.hist_length-1):
            cb(time_hist_am_var[i], time_hist[i+1])

        time_hist_am = time_hist_am_var

        # Set up the actual Adams-Moulton step.
        am_sum = emit_adams_integration(
                        cb, name_gen,
                        self.function_family,
                        time_hist_am, history_am,
                        0, t_end)

        ab_sum = emit_adams_integration(
                        cb, name_gen,
                        self.function_family,
                        time_hist_ab, history_ab,
                        0, t_end)

        state_est_pred = self.state + dt_factor * ab_sum
        state_est_corr = self.state + dt_factor * am_sum

        # Build the implicit solve expression.
        from dagrt.expression import collapse_constants
        from pymbolic.mapper.distributor import DistributeMapper as DistMap
        solve_expression = collapse_constants(
                rhs_next_var - self.eval_rhs(self.t + self.dt,
                                                 DistMap()(state_est_corr)),
                list(unknowns) + [self.state],
                cb.assign, cb.fresh_var)
        equations.append(solve_expression)

        # {{{ emit solve if possible

        if unknowns and len(unknowns) == len(equations):
            from leap.implicit import generate_solve
            generate_solve(cb, unknowns, equations,
                           rhs_var_to_unknown, state_est_pred)

        del equations[:]
        knowns.update(unknowns)
        unknowns.clear()

        # }}}

        # Update the state now that we've solved.
        if self.state_filter is not None:
            state_est_pred = self.state_filter(state_est_pred)
            state_est_corr = self.state_filter(state_est_corr)

        # Finish needs to intervene here.
        self.finish(cb, state_est_corr, state_est_pred, history, time_history_data)

    def finish(self, cb, high_est, low_est, hist, time_hist):
        if not self.adaptive:
            cb(self.state, low_est)
            # Rotate history and time history.
            self.rotate_and_yield(cb, hist, time_hist)
        else:
            self.finish_adaptive(cb, high_est, low_est, hist, time_hist)

    def finish_nonadaptive(self, cb, high_order_estimate,
                           low_order_estimate, hist, time_hist):
        if self.use_high_order:
            est = high_order_estimate
        else:
            est = low_order_estimate

        cb(self.state, est)
        # Rotate history and time history.
        self.rotate_and_yield(cb, hist, time_hist)

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

        from leap.rk import ORDER_TO_RK_METHOD_BUILDER
        rk_method = ORDER_TO_RK_METHOD_BUILDER[self.function_family.order]
        rk_coeffs = rk_method.output_coeffs
        stage_coeff_set_names = ("explicit",)
        stage_coeff_sets = {"explicit": rk_method.a_explicit}
        estimate_coeff_set_names = ("main",)
        estimate_coeff_sets = {"main": rk_coeffs}
        rhs_funcs = {"explicit": var("<func>"+self.component_id)}

        # Traverse RK stage loop of appropriate order and update state.
        rk = rk_method(self.component_id, self.state_filter_name)
        cb = rk.generate_butcher_init(cb, stage_coeff_set_names,
                                      stage_coeff_sets, rhs_funcs,
                                      estimate_coeff_set_names,
                                      estimate_coeff_sets)
        cb, rhss, est_vars = rk.generate_butcher_primary(cb, stage_coeff_set_names,
                                                         stage_coeff_sets, rhs_funcs,
                                                         estimate_coeff_set_names,
                                                         estimate_coeff_sets)

        # Assign the value of the new state.
        cb(self.state, est_vars[0])

    def determine_bootstrap_length(self):

        # In the explicit case, this is always
        # equal to history length.
        bootstrap_length = self.hist_length

        return bootstrap_length

# }}}


# vim: fdm=marker
