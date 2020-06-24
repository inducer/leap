"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = """
Copyright (C) 2007-2013 Andreas Kloeckner
Copyright (C) 2014, 2015 Matt Wala
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

import six
import numpy as np
from leap import MethodBuilder, TwoOrderAdaptiveMethodBuilderMixin
from dagrt.language import CodeBuilder, DAGCode

from pymbolic import var


__doc__ = """
.. data:: ORDER_TO_RK_METHOD_BUILDER

    A dictionary mapping desired order of accuracy to a corresponding RK method
    builder.

.. autoclass:: ForwardEulerMethodBuilder
.. autoclass:: BackwardEulerMethodBuilder
.. autoclass:: MidpointMethodBuilder
.. autoclass:: HeunsMethodBuilder
.. autoclass:: RK3MethodBuilder
.. autoclass:: RK4MethodBuilder
.. autoclass:: RK5MethodBuilder

Low-Storage Methods
-------------------

.. autoclass:: LSRK4MethodBuilder

Adaptive/Embedded Methods
-------------------------

.. autoclass:: ODE23MethodBuilder
.. autoclass:: ODE45MethodBuilder

"""


# {{{ utilities

def _truncate_final_zeros(array):
    if not array:
        return array

    index = len(array) - 1
    while array[index] == 0 and index >= 0:
        index -= 1
    return array[:index + 1]


def _is_first_stage_same_as_last_stage(c, coeff_set):
    return (
            c
            and c[0] == 0
            and c[-1] == 1
            and not _truncate_final_zeros(coeff_set[0]))


def _is_last_stage_same_as_output(c, coeff_sets, output_stage_coefficients):
    return (
            c

            and all(
                coeff_set[-1]
                for coeff_set in six.itervalues(coeff_sets))

            and output_stage_coefficients

            and all(
                _truncate_final_zeros(coeff_set[-1])
                ==  # noqa: W504
                _truncate_final_zeros(output_stage_coefficients)
                for coeff_set in six.itervalues(coeff_sets)))

# }}}


# {{{ fully general butcher tableau to code

class ButcherTableauMethodBuilder(MethodBuilder):
    """Infrastructure to generate code from butcher tableaux."""

    @property
    def c(self):
        raise NotImplementedError

    @property
    def a_explicit(self):
        raise NotImplementedError

    @property
    def output_coeffs(self):
        raise NotImplementedError

    @property
    def recycle_last_stage_coeff_set_names(self):
        raise NotImplementedError

    def __init__(self, component_id, state_filter_name=None):

        self.component_id = component_id

        self.dt = var('<dt>')
        self.t = var('<t>')
        self.state = var('<state>' + component_id)

        if state_filter_name is not None:
            self.state_filter = var("<func>" + state_filter_name)
        else:
            self.state_filter = None

    def generate_butcher(self, stage_coeff_set_names, stage_coeff_sets, rhs_funcs,
            estimate_coeff_set_names, estimate_coeff_sets):
        """
        :arg stage_coeff_set_names: a list of names/string identifiers
            for stage coefficient sets
        :arg stage_coeff_sets: a mapping from set names to stage coefficients
        :arg rhs_funcs: a mapping from set names to right-hand-side
            functions
        :arg estimate_coeffs_set_names: a list of names/string identifiers
            for estimate coefficient sets
        :arg estimate_coeffs_sets: a mapping from estimate coefficient set
            names to cofficients.
        """

        from pymbolic import var
        comp = self.component_id

        dt = self.dt
        t = self.t
        state = self.state

        nstages = len(self.c)

        # {{{ check coefficients for plausibility

        for name in stage_coeff_set_names:
            for istage in range(nstages):
                coeff_sum = sum(stage_coeff_sets[name][istage])
                assert abs(coeff_sum - self.c[istage]) < 1e-12, (
                        name, istage, coeff_sum, self.c[istage])

        # }}}

        # {{{ initialization

        last_rhss = {}

        with CodeBuilder(name="initialization") as cb:
            for name in stage_coeff_set_names:
                if (
                        name in self.recycle_last_stage_coeff_set_names
                        and _is_first_stage_same_as_last_stage(
                        self.c, stage_coeff_sets[name])):
                    last_rhss[name] = var("<p>last_rhs_" + name)
                    cb(last_rhss[name], rhs_funcs[name](t=t, **{comp: state}))

        cb_init = cb

        # }}}

        stage_rhs_vars = {}
        rhs_var_to_unknown = {}
        for name in stage_coeff_set_names:
            stage_rhs_vars[name] = [
                    cb.fresh_var('rhs_%s_s%d' % (name, i)) for i in range(nstages)]

            # These are rhss if they are not yet known and pending an implicit solve.
            for i, rhsvar in enumerate(stage_rhs_vars[name]):
                unkvar = cb.fresh_var('unk_%s_s%d' % (name, i))
                rhs_var_to_unknown[rhsvar] = unkvar

        knowns = set()

        # {{{ stage loop

        last_state_est_var = cb.fresh_var("last_state_est")
        last_state_est_var_valid = False

        with CodeBuilder(name="primary") as cb:
            equations = []
            unknowns = set()

            def make_known(v):
                unknowns.discard(v)
                knowns.add(v)

            for istage in range(nstages):
                for name in stage_coeff_set_names:
                    c = self.c[istage]
                    my_rhs = stage_rhs_vars[name][istage]

                    if (
                            name in self.recycle_last_stage_coeff_set_names
                            and istage == 0
                            and _is_first_stage_same_as_last_stage(
                                self.c, stage_coeff_sets[name])):
                        cb(my_rhs, last_rhss[name])
                        make_known(my_rhs)

                    else:
                        is_implicit = False

                        state_increment = 0
                        for src_name in stage_coeff_set_names:
                            coeffs = stage_coeff_sets[src_name][istage]
                            for src_istage, coeff in enumerate(coeffs):
                                rhsval = stage_rhs_vars[src_name][src_istage]
                                if rhsval not in knowns:
                                    unknowns.add(rhsval)
                                    is_implicit = True

                                state_increment += dt * coeff * rhsval

                        state_est = state + state_increment
                        if (self.state_filter is not None
                                and not (
                                    # reusing last output state
                                    c == 0
                                    and all(
                                        len(stage_coeff_sets[src_name][istage]) == 0
                                        for src_name in stage_coeff_set_names))):
                            state_est = self.state_filter(state_est)

                        if is_implicit:
                            rhs_expr = rhs_funcs[name](
                                    t=t + c*dt, **{comp: state_est})

                            from dagrt.expression import collapse_constants
                            solve_expression = collapse_constants(
                                    my_rhs - rhs_expr,
                                    list(unknowns) + [self.state],
                                    cb.assign, cb.fresh_var)
                            equations.append(solve_expression)

                            if istage + 1 == nstages:
                                last_state_est_var_valid = False

                        else:
                            if istage + 1 == nstages:
                                cb(last_state_est_var, state_est)
                                state_est = last_state_est_var
                                last_state_est_var_valid = True

                            rhs_expr = rhs_funcs[name](
                                    t=t + c*dt, **{comp: state_est})

                            cb(my_rhs, rhs_expr)
                            make_known(my_rhs)

                    # {{{ emit solve if possible

                    if unknowns and len(unknowns) == len(equations):
                        # got a square system, let's solve
                        assignees = [unk.name for unk in unknowns]

                        from pymbolic import substitute
                        subst_dict = dict(
                                (rhs_var.name, rhs_var_to_unknown[rhs_var])
                                for rhs_var in unknowns)

                        cb.assign_implicit(
                                assignees=assignees,
                                solve_components=[
                                    rhs_var_to_unknown[unk].name
                                    for unk in unknowns],
                                expressions=[
                                    substitute(eq, subst_dict)
                                    for eq in equations],

                                # TODO: Could supply a starting guess
                                other_params={
                                    "guess": state},
                                solver_id="solve")

                        del equations[:]
                        knowns.update(unknowns)
                        unknowns.clear()

                    # }}}

            # Compute solution estimates.
            estimate_vars = [
                    cb.fresh_var("est_"+name)
                    for name in estimate_coeff_set_names]

            for iest, name in enumerate(estimate_coeff_set_names):
                out_coeffs = estimate_coeff_sets[name]

                if (
                        last_state_est_var_valid
                        and  # noqa: W504
                        _is_last_stage_same_as_output(self.c,
                            stage_coeff_sets, out_coeffs)):
                    state_est = last_state_est_var

                else:
                    state_increment = 0
                    for src_name in stage_coeff_set_names:
                        state_increment += sum(
                                    coeff * stage_rhs_vars[src_name][src_istage]
                                    for src_istage, coeff in enumerate(out_coeffs))

                    state_est = state + dt*state_increment

                    if self.state_filter is not None:
                        state_est = self.state_filter(state_est)

                cb(
                        estimate_vars[iest],
                        state_est)

            # This updates <t>.
            self.finish(cb, estimate_coeff_set_names, estimate_vars)

            # These updates have to happen *after* finish because before we
            # don't yet know whether finish will accept the new state.
            for name in stage_coeff_set_names:
                if (
                        name in self.recycle_last_stage_coeff_set_names
                        and _is_first_stage_same_as_last_stage(
                            self.c, stage_coeff_sets[name])):
                    cb(last_rhss[name], stage_rhs_vars[name][-1])

        cb_primary = cb

        # }}}

        return DAGCode(
                phases={
                    "initial": cb_init.as_execution_phase(next_phase="primary"),
                    "primary": cb_primary.as_execution_phase(next_phase="primary")
                    },
                initial_phase="initial")

    def finish(self, cb, estimate_names, estimate_vars):
        cb(self.state, estimate_vars[0])
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')
        cb(self.t, self.t + self.dt)

# }}}


# {{{ simple butcher tableau methods

class SimpleButcherTableauMethodBuilder(ButcherTableauMethodBuilder):
    def __init__(self, component_id, state_filter_name=None,
            rhs_func_name=None):
        super(SimpleButcherTableauMethodBuilder, self).__init__(
                component_id=component_id,
                state_filter_name=state_filter_name)

        if rhs_func_name is None:
            rhs_func_name = "<func>"+self.component_id
        self.rhs_func_name = rhs_func_name

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """
        return self.generate_butcher(
                stage_coeff_set_names=("explicit",),
                stage_coeff_sets={
                    "explicit": self.a_explicit},
                rhs_funcs={"explicit": var(self.rhs_func_name)},
                estimate_coeff_set_names=("main",),
                estimate_coeff_sets={
                    "main": self.output_coeffs,
                    })


class ImplicitButcherTableauMethodBuilder(ButcherTableauMethodBuilder):
    def __init__(self, component_id, state_filter_name=None,
            rhs_func_name=None):
        super(SimpleButcherTableauMethodBuilder, self).__init__(
                component_id=component_id,
                state_filter_name=state_filter_name)

        if rhs_func_name is None:
            rhs_func_name = "<func>"+self.component_id
        self.rhs_func_name = rhs_func_name

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """
        return self.generate_butcher(
                stage_coeff_set_names=("implicit",),
                stage_coeff_sets={
                    "implicit": self.a_implicit},
                rhs_funcs={"implicit": var(self.rhs_func_name)},
                estimate_coeff_set_names=("main",),
                estimate_coeff_sets={
                    "main": self.output_coeffs,
                    })


class ForwardEulerMethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    """
    c = (0,)

    a_explicit = (
            (),
            )

    output_coeffs = (1,)

    recycle_last_stage_coeff_set_names = ()


class BackwardEulerMethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    """
    c = (1,)

    a_implicit = (
            (1,),
            )

    output_coeffs = (1,)

    recycle_last_stage_coeff_set_names = ()


class MidpointMethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    """
    c = [0, 1/2]

    a_explicit = (
            (),
            (1/2,),
            )

    output_coeffs = (0, 1)

    recycle_last_stage_coeff_set_names = ()


class HeunsMethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    """
    c = [0, 1]

    a_explicit = (
            (),
            (1,),
            )

    output_coeffs = (1/2, 1/2)

    recycle_last_stage_coeff_set_names = ()


class RK3MethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    Source: J. C. Butcher, Numerical MethodBuilders for Ordinary Differential
    Equations, 2nd ed., pages 94 - 99

    .. automethod:: __init__
    .. automethod:: generate
    """

    c = (0, 2/3, 2/3)

    a_explicit = (
        (),
        (2/3,),
        (1/3, 1/3),
        )

    output_coeffs = (1/4, 0, 3/4)
    recycle_last_stage_coeff_set_names = ()


class RK4MethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    """
    c = (0, 1/2, 1/2, 1)

    a_explicit = (
            (),
            (1/2,),
            (0, 1/2,),
            (0, 0, 1,),
            )

    output_coeffs = (1/6, 1/3, 1/3, 1/6)

    recycle_last_stage_coeff_set_names = ()


class RK5MethodBuilder(SimpleButcherTableauMethodBuilder):
    """
    Source: J. C. Butcher, Numerical MethodBuilders for Ordinary Differential
    Equations, 2nd ed., pages 94 - 99
    """

    c = (0, 1/4, 1/4, 1/2, 3/4, 1,)

    a_explicit = (
        (),
        (1/4,),
        (1/8, 1/8),
        (0, 0, 1/2),
        (3/16, -3/8, 3/8, 9/16),
        (-3/7, 8/7, 6/7, -12/7, 8/7),
    )

    output_coeffs = (7/90, 0, 32/90, 12/90, 32/90, 7/90)

    recycle_last_stage_coeff_set_names = ()


class DIRK2MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
            Methods for Ordinary Differential Equations. A Review
            pp. 72, eqn 221
    """

    x = (2 - np.sqrt(2))/2

    c = (x, 1)

    a_implicit = (
            (x,),
            (1 - x, x,),
            )

    output_coeffs = (1 - x, x)

    recycle_last_stage_coeff_set_names = ()


class DIRK3MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
            Methods for Ordinary Differential Equations. A Review
            pp. 77, eqn 229 & eqn 230
    """

    x = 0.4358665215

    c = (x, (1 + x)/2, 1)

    a_implicit = (
            (x,),
            ((1 - x)/2, x,),
            (-3*x**2/2 + 4*x - 0.25, 3*x**2/2 - 5*x + 5/4, x,),
            )

    output_coeffs = (-3*x**2/2 + 4*x - 0.25, 3*x**2/2 - 5*x + 5/4, x)

    recycle_last_stage_coeff_set_names = ()


class DIRK4MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
            Methods for Ordinary Differential Equations. A Review
            pp. 78, eqn 232
    """

    x1 = 1.06858

    c = (x1, 1/2, 1 - x1)

    a_implicit = (
            (x1,),
            (1/2 - x1, x1,),
            (2*x1, 1 - 4*x1, x1),
            )

    output_coeffs = (1/(6*(1 - 2*x1)**2), (3*(1 - 2*x1)**2 - 1)/(3*(1 - 2*x1)**2),
                     1/(6*(1 - 2*x1)**2))

    recycle_last_stage_coeff_set_names = ()


class DIRK5MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    .. automethod:: __init__
    .. automethod:: generate
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
            Methods for Ordinary Differential Equations. A Review
            pp. 98, Table 24
    """

    c = (4024571134387/14474071345096, 5555633399575/5431021154178,
         5255299487392/12852514622453, 3/20, 10449500210709/14474071345096)

    a_implicit = (
            (4024571134387/14474071345096,),
            (9365021263232/12572342979331, 4024571134387/14474071345096,),
            (2144716224527/9320917548702, -397905335951/4008788611757,
             4024571134387/14474071345096,),
            (-291541413000/6267936762551, 226761949132/4473940808273,
             -1282248297070/9697416712681, 4024571134387/14474071345096,),
            (-2481679516057/4626464057815, -197112422687/6604378783090,
             3952887910906/9713059315593, 4906835613583/8134926921134,
             4024571134387/14474071345096,),
            )

    output_coeffs = (-2522702558582/12162329469185, 1018267903655/12907234417901,
                     4542392826351/13702606430957, 5001116467727/12224457745473,
                     1509636094297/3891594770934)

    recycle_last_stage_coeff_set_names = ()


ORDER_TO_RK_METHOD_BUILDER = {
        1: ForwardEulerMethodBuilder,
        2: MidpointMethodBuilder,
        3: RK3MethodBuilder,
        4: RK4MethodBuilder,
        5: RK5MethodBuilder,
        }

IMPLICIT_ORDER_TO_RK_METHOD_BUILDER = {
        1: BackwardEulerMethodBuilder,
        2: DIRK2MethodBuilder,
        3: DIRK3MethodBuilder,
        4: DIRK4MethodBuilder,
        5: DIRK4MethodBuilder,
        }

# }}}


# {{{ Embedded Runge-Kutta schemes base class

class EmbeddedButcherTableauMethodBuilder(
        ButcherTableauMethodBuilder, TwoOrderAdaptiveMethodBuilderMixin):
    """
    User-supplied context:
        <state> + component_id: The value that is integrated
        <func> + component_id: The right hand side function
    """

    @property
    def high_order_coeffs(self):
        raise NotImplementedError

    @property
    def low_order_coeffs(self):
        raise NotImplementedError

    def __init__(self, component_id, use_high_order=True, state_filter_name=None,
            atol=0, rtol=0, max_dt_growth=None, min_dt_shrinkage=None):
        ButcherTableauMethodBuilder.__init__(
                self,
                component_id=component_id,
                state_filter_name=state_filter_name)

        TwoOrderAdaptiveMethodBuilderMixin.__init__(
                self,
                atol=atol,
                rtol=rtol,
                max_dt_growth=max_dt_growth,
                min_dt_shrinkage=min_dt_shrinkage)

        self.use_high_order = use_high_order

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """
        if self.use_high_order:
            estimate_names = ("high_order", "low_order")
        else:
            estimate_names = ("low_order", "high_order")

        return self.generate_butcher(
                stage_coeff_set_names=("explicit",),
                stage_coeff_sets={
                    "explicit": self.a_explicit},
                rhs_funcs={"explicit": var("<func>"+self.component_id)},
                estimate_coeff_set_names=estimate_names,
                estimate_coeff_sets={
                    "high_order": self.high_order_coeffs,
                    "low_order": self.low_order_coeffs
                    })

    def finish(self, cb, estimate_coeff_set_names, estimate_vars):
        if not self.adaptive:
            super(EmbeddedButcherTableauMethodBuilder, self).finish(
                    cb, estimate_coeff_set_names, estimate_vars)
        else:
            high_est = estimate_vars[
                    estimate_coeff_set_names.index("high_order")]
            low_est = estimate_vars[
                    estimate_coeff_set_names.index("low_order")]
            self.finish_adaptive(cb, high_est, low_est)

    def finish_nonadaptive(self, cb, high_order_estimate, low_order_estimate):
        if self.use_high_order:
            est = high_order_estimate
        else:
            est = low_order_estimate

        cb(self.state, est)
        cb.yield_state(self.state, self.component_id, self.t + self.dt, 'final')
        cb(self.t, self.t + self.dt)

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23MethodBuilder(EmbeddedButcherTableauMethodBuilder):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    http://dx.doi.org/10.1016/0893-9659(89)90079-7

    .. automethod:: __init__
    .. automethod:: generate
    """

    c = [0, 1/2, 3/4, 1]

    a_explicit = [
            [],
            [1/2],
            [0, 3/4],
            [2/9, 1/3, 4/9],
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

    recycle_last_stage_coeff_set_names = ("explicit",)

# }}}


# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta

class ODE45MethodBuilder(EmbeddedButcherTableauMethodBuilder):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    http://dx.doi.org/10.1016/0771-050X(80)90013-3.

    .. automethod:: __init__
    .. automethod:: generate
    """

    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a_explicit = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

    recycle_last_stage_coeff_set_names = ("explicit",)

# }}}


# {{{ Carpenter/Kennedy low-storage fourth-order Runge-Kutta

class LSRK4MethodBuilder(MethodBuilder):
    """A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin MethodBuilders p.64
    or
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994

    .. automethod:: __init__
    .. automethod:: generate
    """

    _RK4A = [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
            ]

    _RK4B = [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
            ]

    _RK4C = [
            0.0,
            1432997174477/9575080441755,
            2526269341429/6820363962896,
            2006345519317/3224310063776,
            2802321613138/2924317926251,
            #1,
            ]
    coeffs = np.array([_RK4A, _RK4B, _RK4C]).T

    adaptive = False

    def __init__(self, component_id, state_filter_name=None, rhs_func_name=None):
        """
        :arg component_id: an identifier to be used for the single state
            component supported.
        """

        # Set up variables.
        from pymbolic import var

        self.component_id = component_id

        if state_filter_name is not None:
            self.state_filter = var("<func>" + state_filter_name)
        else:
            self.state_filter = None

        if rhs_func_name is None:
            rhs_func_name = "<func>" + component_id

        self.rhs_func_name = rhs_func_name

    def generate(self):
        """
        :returns: :class:`dagrt.language.DAGCode`
        """
        comp_id = self.component_id

        from pymbolic import var
        dt = var("<dt>")
        t = var("<t>")
        residual = var("<p>residual_" + comp_id)
        state = var("<state>" + comp_id)
        rhs_func = var(self.rhs_func_name)

        with CodeBuilder("initialization") as cb:
            cb(residual, 0)

        cb_init = cb

        # Primary.

        rhs_val = var("rhs_val")

        with CodeBuilder("primary") as cb:
            # https://github.com/PyCQA/pylint/issues/3387
            for a, b, c in self.coeffs:  # pylint: disable=not-an-iterable
                cb(rhs_val, rhs_func(t=t + c*dt, **{comp_id: state}))
                cb(residual, a*residual + dt*rhs_val)
                new_state_expr = state + b * residual

                if self.state_filter is not None:
                    new_state_expr = self.state_filter(**{comp_id: new_state_expr})

                cb(state, new_state_expr)

            cb.yield_state(state, comp_id, t + dt, 'final')
            cb(t, t + dt)

        cb_primary = cb

        from dagrt.language import DAGCode
        return DAGCode(
                phases={
                    "initial": cb_init.as_execution_phase(next_phase="primary"),
                    "primary": cb_primary.as_execution_phase(next_phase="primary")
                    },
                initial_phase="initial")

# }}}


# vim: foldmethod=marker
