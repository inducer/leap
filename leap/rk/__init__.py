"""Runge-Kutta ODE timestepper."""


__copyright__ = """
Copyright (C) 2007-2013 Andreas Kloeckner
Copyright (C) 2014, 2015 Matt Wala
Copyright (C) 2020 Cory Mikida
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
from leap import MethodBuilder, TwoOrderAdaptiveMethodBuilderMixin
from dagrt.language import CodeBuilder, DAGCode

from pymbolic import var


__doc__ = """
.. data:: ORDER_TO_RK_METHOD_BUILDER

    A dictionary mapping desired order of accuracy to a corresponding RK method
    builder.

.. data:: IMPLICIT_ORDER_TO_RK_METHOD_BUILDER

    A dictionary mapping desired order of accuracy to a corresponding implicit
    RK method builder.

.. autoclass:: ForwardEulerMethodBuilder
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

Strong Stability Preserving (SSP) Methods
-----------------------------------------

.. autoclass:: SSPRKMethodBuilder
.. autoclass:: SSPRK22MethodBuilder
.. autoclass:: SSPRK33MethodBuilder

Implicit Methods
-----------------------------------------

.. autoclass:: BackwardEulerMethodBuilder
.. autoclass:: DIRK2MethodBuilder
.. autoclass:: DIRK3MethodBuilder
.. autoclass:: DIRK4MethodBuilder
.. autoclass:: DIRK5MethodBuilder
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
                for coeff_set in coeff_sets.values())

            and output_stage_coefficients

            and all(
                _truncate_final_zeros(coeff_set[-1])
                ==  # noqa: W504
                _truncate_final_zeros(output_stage_coefficients)
                for coeff_set in coeff_sets.values()))

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
    def a_implicit(self):
        raise NotImplementedError

    @property
    def output_coeffs(self):
        raise NotImplementedError

    @property
    def recycle_last_stage_coeff_set_names(self):
        raise NotImplementedError

    def __init__(self, component_id, state_filter_name=None):

        self.component_id = component_id

        self.dt = var("<dt>")
        self.t = var("<t>")
        self.state = var("<state>" + component_id)

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
        # {{{ check coefficients for plausibility

        for name in stage_coeff_set_names:
            for istage in range(len(self.c)):
                coeff_sum = sum(stage_coeff_sets[name][istage])
                assert abs(coeff_sum - self.c[istage]) < 1e-12, (
                        name, istage, coeff_sum, self.c[istage])

        # }}}

        cb_init = CodeBuilder(name="initialization")
        cb_init = self.generate_butcher_init(cb_init, stage_coeff_set_names,
                                              stage_coeff_sets, rhs_funcs,
                                              estimate_coeff_set_names,
                                              estimate_coeff_sets)
        cb_primary = CodeBuilder(name="primary")
        cb_primary, stage_rhs_vars, estimate_vars,  = self.generate_butcher_primary(
                                              cb_primary, stage_coeff_set_names,
                                              stage_coeff_sets, rhs_funcs,
                                              estimate_coeff_set_names,
                                              estimate_coeff_sets)
        cb_primary = self.generate_butcher_finish(cb_primary, stage_coeff_set_names,
                                              stage_coeff_sets, rhs_funcs,
                                              estimate_coeff_set_names,
                                              estimate_coeff_sets,
                                              stage_rhs_vars,
                                              estimate_vars)
        return DAGCode(
                phases={
                    "initial": cb_init.as_execution_phase(next_phase="primary"),
                    "primary": cb_primary.as_execution_phase(next_phase="primary")
                    },
                initial_phase="initial")

    # {{{ initialization

    def generate_butcher_init(self, cb, stage_coeff_set_names,
            stage_coeff_sets, rhs_funcs, estimate_coeff_set_names,
            estimate_coeff_sets):
        self.last_rhss = {}
        from pymbolic import var
        for name in stage_coeff_set_names:
            if (
                    name in self.recycle_last_stage_coeff_set_names
                    and _is_first_stage_same_as_last_stage(
                    self.c, stage_coeff_sets[name])):
                self.last_rhss[name] = var("<p>last_rhs_" + name)
                cb(self.last_rhss[name], rhs_funcs[name](
                    t=self.t, **{self.component_id: self.state}))

        cb_init = cb

        return cb_init

    # }}}

    # {{{ stage loop

    def generate_butcher_primary(self, cb, stage_coeff_set_names,
            stage_coeff_sets, rhs_funcs, estimate_coeff_set_names,
            estimate_coeff_sets):

        last_state_est_var = cb.fresh_var("last_state_est")
        last_state_est_var_valid = False

        equations = []
        unknowns = set()

        comp = self.component_id
        nstages = len(self.c)
        dt = self.dt
        t = self.t
        last_rhss = self.last_rhss

        stage_rhs_vars = {}
        rhs_var_to_unknown = {}
        for name in stage_coeff_set_names:
            stage_rhs_vars[name] = [
                    cb.fresh_var(f"rhs_{name}_s{i}") for i in range(len(self.c))]

            # These are rhss if they are not yet known and pending an implicit solve.
            for i, rhsvar in enumerate(stage_rhs_vars[name]):
                unkvar = cb.fresh_var(f"unk_{name}_s{i}")
                rhs_var_to_unknown[rhsvar] = unkvar

        knowns = set()

        for istage in range(len(self.c)):
            for name in stage_coeff_set_names:
                c = self.c[istage]
                my_rhs = stage_rhs_vars[name][istage]

                if (
                        name in self.recycle_last_stage_coeff_set_names
                        and istage == 0
                        and _is_first_stage_same_as_last_stage(
                            self.c, stage_coeff_sets[name])):
                    cb(my_rhs, last_rhss[name])
                    knowns.add(my_rhs)

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

                    state_est = self.state + state_increment
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
                        knowns.add(my_rhs)

                # {{{ emit solve if we have any unknowns/equations

                if unknowns and len(unknowns) == len(equations):
                    from leap.implicit import generate_solve
                    generate_solve(cb, unknowns, equations, rhs_var_to_unknown,
                                    self.state)
                elif not unknowns:
                    # we have an explicit Runge-Kutta method
                    pass
                elif len(unknowns) > len(equations):
                    raise ValueError("Runge-Kutta implicit timestep has more "
                            "unknowns than equations")
                elif len(unknowns) < len(equations):
                    raise ValueError("Runge-Kutta implicit timestep has more "
                            "equations than unknowns")
                else:
                    assert False

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

                state_est = self.state + dt*state_increment

                if self.state_filter is not None:
                    state_est = self.state_filter(state_est)

            cb(
                estimate_vars[iest],
                state_est)

        cb_primary = cb

        return cb_primary, stage_rhs_vars, estimate_vars

    # }}}

    # {{{ update state and time

    def generate_butcher_finish(self, cb, stage_coeff_set_names,
            stage_coeff_sets, rhs_funcs, estimate_coeff_set_names,
            estimate_coeff_sets, stage_rhs_vars, estimate_vars):

        last_rhss = self.last_rhss

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

        return cb_primary

    # }}}

    def finish(self, cb, estimate_names, estimate_vars):
        cb(self.state, estimate_vars[0])
        cb.yield_state(self.state, self.component_id, self.t + self.dt, "final")
        cb(self.t, self.t + self.dt)

# }}}


# {{{ simple butcher tableau methods

class SimpleButcherTableauMethodBuilder(ButcherTableauMethodBuilder):
    def __init__(self, component_id, state_filter_name=None,
            rhs_func_name=None):
        super().__init__(
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


ORDER_TO_RK_METHOD_BUILDER = {
        1: ForwardEulerMethodBuilder,
        2: MidpointMethodBuilder,
        3: RK3MethodBuilder,
        4: RK4MethodBuilder,
        5: RK5MethodBuilder,
        }

# }}}


# {{{ implicit butcher tableau methods


class ImplicitButcherTableauMethodBuilder(ButcherTableauMethodBuilder):
    def __init__(self, component_id, state_filter_name=None,
            rhs_func_name=None):
        super().__init__(
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


class BackwardEulerMethodBuilder(ImplicitButcherTableauMethodBuilder):
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


class DIRK2MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
    Methods for Ordinary Differential Equations. A Review
    pp. 72, eqn 221

    .. automethod:: __init__
    .. automethod:: generate
    """

    _x = (2 - np.sqrt(2))/2

    c = (_x, 1)

    a_implicit = (
            (_x,),
            (1 - _x, _x,),
            )

    output_coeffs = (1 - _x, _x)

    recycle_last_stage_coeff_set_names = ()


class DIRK3MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
    Methods for Ordinary Differential Equations. A Review
    pp. 77, eqn 229 & eqn 230

    .. automethod:: __init__
    .. automethod:: generate
    """

    _x = 0.4358665215

    c = (_x, (1 + _x)/2, 1)

    a_implicit = (
            (_x,),
            ((1 - _x)/2, _x,),
            (-3*_x**2/2 + 4*_x - 0.25, 3*_x**2/2 - 5*_x + 5/4, _x,),
            )

    output_coeffs = (-3*_x**2/2 + 4*_x - 0.25, 3*_x**2/2 - 5*_x + 5/4, _x)

    recycle_last_stage_coeff_set_names = ()


class DIRK4MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
    Methods for Ordinary Differential Equations. A Review
    pp. 78, eqn 232

    .. automethod:: __init__
    .. automethod:: generate
    """

    _x1 = 1.06858

    c = (_x1, 1/2, 1 - _x1)

    a_implicit = (
            (_x1,),
            (1/2 - _x1, _x1,),
            (2*_x1, 1 - 4*_x1, _x1),
            )

    output_coeffs = (1/(6*(1 - 2*_x1)**2), (3*(1 - 2*_x1)**2 - 1)/(3*(1 - 2*_x1)**2),
                     1/(6*(1 - 2*_x1)**2))

    recycle_last_stage_coeff_set_names = ()


class DIRK5MethodBuilder(ImplicitButcherTableauMethodBuilder):
    """
    Source: Kennedy & Carpenter: Diagonally Implicit Runge-Kutta
    Methods for Ordinary Differential Equations. A Review
    pp. 98, Table 24

    .. automethod:: __init__
    .. automethod:: generate
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


IMPLICIT_ORDER_TO_RK_METHOD_BUILDER = {
        1: BackwardEulerMethodBuilder,
        2: DIRK2MethodBuilder,
        3: DIRK3MethodBuilder,
        4: DIRK4MethodBuilder,
        5: DIRK5MethodBuilder,
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
            super().finish(
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
        cb.yield_state(self.state, self.component_id, self.t + self.dt, "final")
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

            cb.yield_state(state, comp_id, t + dt, "final")
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


# {{{ Explicit SSP Runge-Kutta methods

class SSPRKMethodBuilder(MethodBuilder):
    r"""Explicit Strong Stability Preserving (SSP) Runge-Kutta Methods.

    The methods are given in the now-standard Shu-Osher form

    .. math::

        \begin{aligned}
        y^{(i)} =\,\, & \sum_{k = 0}^{i - 1}
            \alpha_{ik} y^{(i)} + \Delta t \beta_{ik} f(y^{(i)}), \\
        y^{n + 1} =\,\, & y^{(n)},
        \end{aligned}

    for :math:`i \in \{1, \dots, n\}` and :math:`y^{(0)} = y^n`. For reference,
    see [gst-2001]_.

    .. [gst-2001]

        S. Gottlieb, C.-W. Shu, E. Tadmor, *Strong Stability
        Preserving High-Order Time Discretization Methods*, SIAM, Vol. 43,
        pp. 89-112, 2001.
        https://doi.org/10.1137/S003614450036757X
    """

    def __init__(self, component_id, state_filter_name=None, rhs_func_name=None):
        super().__init__()

        state_filter = state_filter_name
        if state_filter is not None:
            state_filter = var(f"<func>{state_filter_name}")

        if rhs_func_name is None:
            rhs_func_name = component_id

        self.component_id = component_id
        self.dt = var("<dt>")
        self.t = var("<t>")
        self.state = var(f"<state>{component_id}")
        self.state_filter = state_filter
        self.rhs_func = var(f"<func>{rhs_func_name}")

    @property
    def c(self):
        raise NotImplementedError

    @property
    def alpha(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    def generate(self):
        """
        :returns: a :class:`~dagrt.language.DAGCode` that can be used to
            generate code for this method.
        """

        # {{{ check coefficients are explicit

        nstages = len(self.alpha)
        for n in range(1, nstages + 1):
            if len(self.alpha[n - 1]) > n or len(self.beta[n - 1]) > n:
                raise ValueError("only explicit SSP schemes are supported")

        # }}}

        # {{{ primary phase

        comp_id = self.component_id

        with CodeBuilder(name="primary") as cb:
            stages = [self.state] + [
                    cb.fresh_var(f"s{i}") for i in range(nstages)
                    ]

            for i in range(0, nstages):
                states = sum(
                        alpha * stages[j]
                        for j, alpha in enumerate(self.alpha[i])
                        )
                rhss = sum(
                        beta * self.rhs_func(
                            t=self.t + self.c[i] * self.dt,
                            **{comp_id: stages[j]})
                        for j, beta in enumerate(self.beta[i])
                        )

                expr = states + self.dt * rhss
                if self.state_filter is not None:
                    expr = self.state_filter(**{comp_id: expr})

                cb(stages[i + 1], expr)

            # finish
            cb(self.state, stages[-1])
            cb.yield_state(self.state, comp_id, self.t + self.dt, "final")
            cb(self.t, self.t + self.dt)

        # }}}

        return DAGCode(
                phases={
                    "primary": cb.as_execution_phase(next_phase="primary"),
                    },
                initial_phase="primary"
                )


class SSPRK22MethodBuilder(SSPRKMethodBuilder):
    """Second-order SSP Runge-Kutta method from [gst-2001]_ Proposition 4.1.

    .. automethod:: generate
    """

    c = (0, 1)

    alpha = (
            (1,),
            (1/2, 1/2),
            )

    beta = (
            (1,),
            (0, 1/2),
            )


class SSPRK33MethodBuilder(SSPRKMethodBuilder):
    """Third-order SSP Runge-Kutta method from [gst-2001]_ Proposition 4.1.

    .. automethod:: generate
    """

    c = (0, 1, 1/2)

    alpha = (
            (1,),
            (3/4, 1/4),
            (1/3, 0, 2/3),
            )

    beta = (
            (1,),
            (0, 1/4),
            (0, 0, 2/3),
            )

# }}}

# vim: foldmethod=marker
