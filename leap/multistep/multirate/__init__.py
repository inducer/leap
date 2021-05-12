"""Multirate Adams ODE method."""


__copyright__ = """
Copyright (C) 2007-15 Andreas Kloeckner
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

from pytools import Record
from leap import MethodBuilder
from pymbolic import var

from leap.multistep import _linear_comb


__doc__ = """

.. autoclass:: rhs_policy
.. autoclass:: MultiRateHistory
.. autoclass:: MultiRateMultiStepMethodBuilder

.. autoclass:: SchemeExplainerBase
.. autoclass:: TextualSchemeExplainer
"""


# {{{ system description

class rhs_policy:  # noqa
    """
    .. attribute:: late
    .. attribute:: early
    .. attribute:: early_and_late
    """
    late = 0
    early = 1
    early_and_late = 2


class MultiRateHistory(Record):
    """
    .. automethod:: __init__
    """
    def __init__(self, interval, func_name, arguments, order=None,
            rhs_policy=rhs_policy.late, invalidate_computed_state=False,
            hist_length=None):
        """
        :arg interval: An integer indicating the interval (relative to the
            smallest available timestep) at which this history is to be
            updated.
            (where each update will typically involve a call to *func_name*)
        :arg arguments: A tuple of component names
            (see :class:`MultiRateMultiStepMethodBuilder`)
            which are passed to this right-hand side function.
        :arg order: The Adams approximation order to be used for this
            history, or None if the method default is to be used.
        :arg rhs_policy: One of the constants in :class:`rhs_policy`
        :arg invalidate_dependent_state: Whether evaluating this
            right-hand side should force a recomputation of any
            state that depended upon now-superseded state.
        :arg hist_length: history length.  If greater than order, we use a
            least-squares solve rather than a linear solve to obtain the Adams
            coefficients for this history
        """
        super().__init__(
                interval=interval,
                func_name=func_name,
                arguments=arguments,
                order=order,
                rhs_policy=rhs_policy,
                invalidate_computed_state=invalidate_computed_state,
                hist_length=hist_length)

    @property
    def history_length(self):
        return self.hist_length


class RHS(MultiRateHistory):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("RHS is deprecated--use MultiRateHistory instead",
                DeprecationWarning, stacklevel=2)

        super().__init__(*args, **kwargs)

# }}}


# {{{ topological sort of rhss

def _topologically_sort_comp_names_and_rhss(component_names, rhss):
    # This routine is O(n^2) in a few spots, assuming the input is small.
    # Acceleration to amortized O(n) using sets/dicts should be easy if needed.

    result_component_names = []

    deps = {
            cname: frozenset(dep_cname
                for mrh in rhs
                for dep_cname in mrh.arguments if mrh in component_names)
            for cname, rhs in zip(component_names, rhss)
            }

    def add(cname):
        if cname in result_component_names:
            return

        for dep in deps[cname]:
            add(dep)

        if cname in result_component_names:
            raise ValueError(f"Component '{cname}' (directly or indirectly) "
                    "depends on itself in system description. This is not allowed.")

        result_component_names.append(cname)

    for cname in component_names:
        add(cname)

    return result_component_names, [
            rhss[component_names.index(cname)] for cname in result_component_names]

# }}}


class InconsistentHistoryError:
    pass


# {{{ method

class MultiRateMultiStepMethodBuilder(MethodBuilder):
    """Simultaneously timesteps multiple parts of an ODE system,
    each with adjustable orders, rates, and dependencies.

    Considerably generalizes [GearWells]_.

    .. [GearWells] C.W. Gear and D.R. Wells, "Multirate linear multistep methods,"
         BIT Numerical Mathematics,  vol. 24, Dec. 1984,pg. 484-502.

    .. automethod:: __init__
    .. automethod:: generate
    """

    # {{{ constructor

    def __init__(self, default_order, system_description,
            state_filter_names=None,
            component_arg_names=None,
            static_dt=False,
            hist_consistency_threshold=None,
            early_hist_consistency_threshold=None):

        """
        :arg default_order: The order to be used for right-hand sides
            where no differing order is specified.
        :arg system_description: A tuple of the form::

                (
                    ("dt", "fast",
                    "=", MultiRateHistory(1, "<func>f1", ("fast", "slow", "dep")),
                    MultiRateHistory(3, "<func>f2", ("slow", "dep")),
                    ),
                    ("dt", "slow",
                    "=", MultiRateHistory(3, "<func>f3", ("fast", "slow", "dep")),
                    ),
                    ("dep",
                    "=", MultiRateHistory(3, "<func>f4", ("slow")),
                    ),
                )

            i.e. the outermost tuple represents a list of state components.
            These can either be ODEs (in which case the first string in the
            tuple is ``"dt"``, followed by the component name, or
            computed/dependent state, in which case the first string in the
            tuple is just the name of the computed piece of state.

            The right-hand-side of each tuple (after the intervening ``"="``
            string) consists of one or more instances of
            :class:`MultiRateHistory` describing the rate at which evaluations
            of the given functions should be stored (along with other
            parameters that can be configured in :class:`MultiRateHistory`).
            The right-hand sides are combined additively.

            Computed state components may not (directly) or indirectly depend
            on themselves. The result will be undefined.

        :arg state_filter_names: a dictionary mapping (non-ODE) component
            names from the system description to the names of "state filter"
            functions (as in functions that receive this particular component
            state and return a potentially modified version thereof).
        :arg component_arg_names: A tuple of names of the components
            to be used as keywords for passing arguments to the right
            hand sides in *rhss*.
        :arg static_dt: If *True*, changing the timestep in between steps
            during time integration is not allowed.
        """
        super().__init__()

        # Variables
        from pymbolic import var

        self.t = var("<t>")
        self.dt = var("<dt>")
        self.bootstrap_step = var("<p>bootstrap_step")

        # {{{ process system_description

        ode_component_names = []
        ode_rhss = []
        non_ode_component_names = []
        non_ode_rhss = []
        is_ode_component = {}

        if not isinstance(system_description, tuple):
            raise TypeError("'system_description' must be a tuple")

        for irow, row in enumerate(system_description):
            if not isinstance(row, tuple):
                raise TypeError(f"row {irow + 1} (1-based) of 'system_description' "
                        "must be a tuple")

            try:
                eq_index = row.index("=")
            except ValueError:
                raise ValueError(f"row {irow + 1} (1-based) of 'system_description' "
                        "must contain an equal sign")

            is_ode = eq_index == 2

            if is_ode:
                if row[0] != "dt":
                    raise ValueError(f"row {irow + 1} (1-based) of "
                            "'system_description' must have 'dt' as first "
                            "element if describing an ODE")
                comp_name = row[1]

                ode_component_names.append(comp_name)
                ode_rhss.append(row[eq_index+1:])

            elif eq_index == 1:
                comp_name = row[0]

                non_ode_component_names.append(comp_name)
                non_ode_rhss.append(row[eq_index+1:])

            else:
                raise ValueError(f"row {irow + 1} (1-based) of "
                        "'system_description' has equal sign in unexpected location")

            is_ode_component[comp_name] = is_ode

        # Top. sort is required by RK bootstrap.
        non_ode_component_names, non_ode_rhss = \
                _topologically_sort_comp_names_and_rhss(
                        non_ode_component_names, non_ode_rhss)

        # RK bootstrap below relies on ordering: non-ODE, then ODE.
        component_names = non_ode_component_names + ode_component_names
        rhss = non_ode_rhss + ode_rhss

        del non_ode_component_names
        del non_ode_rhss
        del ode_component_names
        del ode_rhss

        # }}}

        # {{{ process state filters

        if state_filter_names is None:
            state_filter_names = {}

        for comp_name, _sfname in state_filter_names.items():
            if comp_name not in component_names:
                raise ValueError(f"component name '{comp_name}' in "
                    "'state_filter_names' not known")

            if not is_ode_component[comp_name]:
                raise ValueError(f"component name '{comp_name}' in "
                    "'state_filter_names' is a non-ODE component, "
                    "which is not allowed")

        self.state_filters = {
                comp_name: var("<func>" + sfname)
                for comp_name, sfname in state_filter_names.items()
                if sfname is not None}

        # }}}

        # {{{ plug default order, history length into rhss

        new_rhss = []
        for component_rhss in rhss:
            new_component_rhss = []
            for rhs in component_rhss:
                order = rhs.order
                hist_length = rhs.hist_length
                if order is None:
                    order = default_order
                if hist_length is None:
                    hist_length = order

                new_component_rhss.append(rhs.copy(order=order,
                    hist_length=hist_length))

            new_rhss.append(tuple(new_component_rhss))

        self.rhss = new_rhss
        del new_rhss
        del rhss

        # }}}

        self.component_names = component_names
        self.is_ode_component = is_ode_component

        if component_arg_names is None:
            component_arg_names = component_names

        self.comp_name_to_kwarg_name = dict(
                zip(component_names, component_arg_names))

        self.max_order = max(rhs.order
                for component_rhss in self.rhss
                for rhs in component_rhss)

        self.max_hist_length = max(rhs.hist_length
                for component_rhss in self.rhss
                for rhs in component_rhss)

        # {{{ process intervals

        intervals = tuple(rhs.interval
                for component_rhss in self.rhss
                for rhs in component_rhss)

        interval_gcd = gcd(intervals)
        if interval_gcd != 1:
            raise ValueError(
                    "integration intervals must be relatively prime: found "
                    "intervals {intervals} with common factor {interval_gcd}")

        self.nsubsteps = lcm(intervals)

        # }}}

        self.static_dt = static_dt
        self.hist_consistency_threshold = hist_consistency_threshold
        if isinstance(early_hist_consistency_threshold, str):
            from dagrt.expression import parse
            early_hist_consistency_threshold = parse(
                    early_hist_consistency_threshold)
        self.early_hist_consistency_threshold = early_hist_consistency_threshold

        if not self.static_dt:
            self.time_vars = {}
        self.history_vars = {}

        for comp_name, component_rhss in zip(self.component_names, self.rhss):
            for irhs, rhs in enumerate(component_rhss):
                key = comp_name, irhs

                # These are organized latest-last.
                t_vars = []
                hist_vars = []
                for past in range(rhs.history_length):
                    t_vars.insert(0, var(
                        f"<p>t_{comp_name}_rhs{irhs}_hist_{past}_ago"))
                    hist_vars.insert(0, var(
                        f"<p>hist_{comp_name}_rhs{irhs}_hist_{past}_ago"))

                if not self.static_dt:
                    self.time_vars[key] = t_vars

                self.history_vars[key] = hist_vars

        self.state_vars = tuple(
                var("<state>" + comp_name) for comp_name in self.component_names)

    # }}}

    def emit_initialization(self, cb):
        """Initialize method variables."""

        cb(self.bootstrap_step, 0)

    # {{{ rk bootstrap: step

    def emit_small_rk_step(self, cb, name_prefix, name_gen, rhss_on_entry):
        """Emit a single step of an RK method."""

        from leap.rk import ORDER_TO_RK_METHOD_BUILDER
        rk_method = ORDER_TO_RK_METHOD_BUILDER[self.max_order]
        rk_tableau = tuple(zip(rk_method.c, rk_method.a_explicit))
        rk_coeffs = rk_method.output_coeffs

        def make_stage_history(prefix):
            return [var(prefix + "_stage" + str(i)) for i in range(len(rk_tableau))]

        stage_rhss = {}
        for comp_name, component_rhss in zip(self.component_names, self.rhss):
            if not self.is_ode_component[comp_name]:
                continue

            for irhs in range(len(component_rhss)):
                stage_rhss[comp_name, irhs] = make_stage_history(
                        "{name_prefix}_rk_{comp_name}_rhs{irhs}"
                        .format(
                            name_prefix=name_prefix,
                            comp_name=comp_name,
                            irhs=irhs))

        for istage, (c, coeffs) in enumerate(rk_tableau):
            if len(coeffs) == 0:
                assert c == 0
                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):
                    if not self.is_ode_component[comp_name]:
                        continue

                    for irhs in range(len(component_rhss)):
                        cb(stage_rhss[comp_name, irhs][istage],
                                rhss_on_entry[comp_name, irhs])

            else:
                component_state_ests = {}

                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):
                    if not self.is_ode_component[comp_name]:
                        continue

                    contribs = []
                    for irhs in range(len(component_rhss)):
                        state_contrib_var = var(
                                name_gen(
                                    "state_contrib_{comp_name}_rhs{irhs}"
                                    .format(comp_name=comp_name, irhs=irhs)))

                        contribs.append(state_contrib_var)

                        cb(state_contrib_var,
                                _linear_comb(coeffs, stage_rhss[comp_name, irhs]))

                    state_var = var(
                            name_gen(
                                "state_{comp_name}_st{istage}"
                                .format(comp_name=comp_name, istage=istage)))

                    state_expr = (
                            var("<state>" + comp_name)
                            + (self.dt/self.nsubsteps) * sum(contribs))
                    if comp_name in self.state_filters:
                        state_expr = self.state_filters[comp_name](state_expr)

                    cb(state_var, state_expr)

                    component_state_ests[comp_name] = state_var

                # At this point, we have all the ODE state estimates evaluated.

                # {{{ evaluate the non-ODE RHSs

                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):
                    if self.is_ode_component[comp_name]:
                        continue

                    contribs = []

                    for rhs in component_rhss:
                        kwargs = {
                                self.comp_name_to_kwarg_name[arg_comp_name]:
                                component_state_ests[arg_comp_name]
                                for arg_comp_name in rhs.arguments}

                        contribs.append(var(rhs.func_name)(
                                    t=self.t + (c/self.nsubsteps) * self.dt,
                                    **kwargs))

                    state_var = var(
                            name_gen(
                                "state_{comp_name}_st{istage}"
                                .format(comp_name=comp_name, istage=istage)))

                    cb(state_var, sum(contribs))

                    component_state_ests[comp_name] = state_var

                # }}}

                # {{{ evaluate the ODE RHSs

                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):

                    if not self.is_ode_component[comp_name]:
                        continue

                    for irhs, rhs in enumerate(component_rhss):
                        kwargs = {
                                self.comp_name_to_kwarg_name[arg_comp_name]:
                                component_state_ests[arg_comp_name]
                                for arg_comp_name in rhs.arguments}
                        cb(stage_rhss[comp_name, irhs][istage],
                                var(rhs.func_name)(
                                    t=self.t + (c/self.nsubsteps) * self.dt,
                                    **kwargs))

                # }}}

        component_state_ests = {}

        for comp_name, component_rhss in zip(self.component_names, self.rhss):

            contribs = []
            for irhs in range(len(component_rhss)):
                if not self.is_ode_component[comp_name]:
                    continue

                state_contrib_var = var(
                        name_gen(
                            "state_contrib_{comp_name}_rhs{irhs}"
                            .format(comp_name=comp_name, irhs=irhs)))

                contribs.append(state_contrib_var)

                cb(state_contrib_var,
                        _linear_comb(rk_coeffs, stage_rhss[comp_name, irhs]))

            state_var = var(
                    name_gen(
                        "state_{comp_name}_final"
                        .format(comp_name=comp_name)))

            state_expr = (
                    var("<state>" + comp_name)
                    + (self.dt/self.nsubsteps) * sum(contribs))
            if comp_name in self.state_filters:
                state_expr = self.state_filters[comp_name](state_expr)

            cb(state_var, state_expr)

            component_state_ests[comp_name] = state_var

        for component_name in self.component_names:
            state = component_state_ests[component_name]
            if self.is_ode_component[component_name]:
                cb.yield_state(
                        state,
                        component_name, self.t + self.dt/self.nsubsteps,
                        "bootstrap")

            cb(var("<state>"+component_name), state)

        cb(self.t, self.t + self.dt/self.nsubsteps)

    # }}}

    # {{{ rk bootstrap: overall control

    def emit_rk_bootstrap(self, cb):
        """Initialize the stepper with an RK method. Return the code that
        computes the startup history."""

        bootstrap_steps = self.max_hist_length - 1

        final_iglobal_substep = bootstrap_steps * self.nsubsteps

        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        for isubstep in range(self.nsubsteps):
            name_prefix = "substep" + str(isubstep)

            current_rhss = {}
            non_ode_states = {}

            # {{{ compute non-ODE current_rhss and states

            for comp_name, component_rhss in zip(
                    self.component_names, self.rhss):
                if self.is_ode_component[comp_name]:
                    continue

                comp_state = 0

                for irhs, rhs in enumerate(component_rhss):
                    rhs_var = var(
                        name_gen(
                            "{name_prefix}_start_{comp_name}_rhs{irhs}"
                            .format(name_prefix=name_prefix, comp_name=comp_name,
                                irhs=irhs)))

                    kwargs = {
                            self.comp_name_to_kwarg_name[arg_comp_name]:
                            var("<state>" + arg_comp_name)
                            for arg_comp_name in rhs.arguments}

                    cb(rhs_var, var(rhs.func_name)(t=self.t, **kwargs))

                    current_rhss[comp_name, irhs] = rhs_var
                    comp_state += rhs_var

                non_ode_state_var = var(
                    name_gen(
                        "{name_prefix}_{comp_name}"
                        .format(name_prefix=name_prefix, comp_name=comp_name)))
                cb(non_ode_state_var, comp_state)

                non_ode_states[comp_name] = non_ode_state_var

            # }}}

            # {{{ compute ODE current_rhss

            for comp_name, component_rhss in zip(
                    self.component_names, self.rhss):
                if not self.is_ode_component[comp_name]:
                    continue

                for irhs, rhs in enumerate(component_rhss):
                    rhs_var = var(
                        name_gen(
                            "{name_prefix}_start_{comp_name}_rhs{irhs}"
                            .format(name_prefix=name_prefix, comp_name=comp_name,
                                irhs=irhs)))

                    def get_state(comp_name):
                        if self.is_ode_component[comp_name]:
                            return var("<state>" + comp_name)
                        else:
                            return non_ode_states[comp_name]

                    kwargs = {
                            self.comp_name_to_kwarg_name[arg_comp_name]:
                            get_state(arg_comp_name)
                            for arg_comp_name in rhs.arguments}

                    cb(rhs_var, var(rhs.func_name)(t=self.t, **kwargs))

                    current_rhss[comp_name, irhs] = rhs_var

            # }}}

            # {{{ collect time/rhs history

            for test_step in range(bootstrap_steps + 1):
                if test_step == bootstrap_steps and isubstep > 0:
                    continue

                test_iglobal_substep = test_step * self.nsubsteps + isubstep

                substeps_from_start = final_iglobal_substep - test_iglobal_substep

                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):
                    for irhs, rhs in enumerate(component_rhss):
                        if (substeps_from_start % rhs.interval == 0
                                and (substeps_from_start // rhs.interval
                                    < rhs.history_length)):

                            intervals_from_start = (
                                    substeps_from_start // rhs.interval)

                            i = rhs.history_length - 1 - intervals_from_start
                            assert i >= 0

                            with cb.if_(self.bootstrap_step, "==", test_step):
                                if not self.static_dt:
                                    cb(self.time_vars[comp_name, irhs][i], self.t)

                                cb(self.history_vars[comp_name, irhs][i],
                                        current_rhss[comp_name, irhs])

            # }}}

            if isubstep == 0:
                with cb.if_(self.bootstrap_step, "==", bootstrap_steps):
                    cb.switch_phase("primary")
                    cb.restart_step()

            self.emit_small_rk_step(cb, name_prefix, name_gen, current_rhss)

        cb(self.bootstrap_step, self.bootstrap_step + 1)

        return cb

    # }}}

    class StateContribExplanation(Record):
        pass

    # {{{ main method generation

    def emit_adams_method(self, cb, explainer):
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        array = var("<builtin>array")

        # {{{ make temporary copies of time/hist_vars

        # maps from (component_name, irhs) to latest-last list of values
        temp_hist_substeps = {}
        temp_time_vars = {}
        temp_hist_vars = {}

        def fill_temp_hist_vars():
            for comp_name, component_rhss in zip(self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    key = comp_name, irhs

                    temp_hist_substeps[key] = list(range(
                        -rhs.interval*(rhs.history_length-1), 1, rhs.interval))

                    if self.static_dt:
                        temp_time_vars[key] = list(
                                rhs.interval*i/self.nsubsteps
                                for i in range(-rhs.history_length+1, 0+1))
                    else:
                        temp_time_vars[key] = self.time_vars[key][:]

                    temp_hist_vars[key] = self.history_vars[key][:]

        fill_temp_hist_vars()

        # }}}

        def log_hist_state():
            explainer.log_hist_state({
                rhs.func_name: (
                    temp_hist_substeps[comp_name, irhs][-rhs.history_length::],
                    [v.name
                        for v in
                        temp_hist_vars[comp_name, irhs][-rhs.history_length::]])
                for comp_name, component_rhss in zip(self.component_names, self.rhss)
                for irhs, rhs in enumerate(component_rhss)})

        log_hist_state()

        # A mapping from component_name to a list of tuples
        # (substep_level, state_var). This mapping is ordered
        # by substep_level.
        computed_states = {
                comp_name: [
                    (0, state_var)
                    ]
                for comp_name, state_var in zip(
                    self.component_names, self.state_vars)}

        # {{{ get_state

        def get_state(comp_name, isubstep):
            states = computed_states[comp_name]

            # {{{ see if we've got that state ready to go

            for istate_substep, state_var in states:
                if istate_substep == isubstep:
                    return state_var

            # }}}

            latest_state_substep, latest_state = states[-1]

            comp_index = self.component_names.index(comp_name)
            rhss = self.rhss[comp_index]

            contribs = []
            contrib_explanations = []

            for irhs, rhs in enumerate(rhss):
                hist_len = rhs.history_length

                relv_hist_substeps = temp_hist_substeps[comp_name, irhs][-hist_len:]
                relv_time_hist = temp_time_vars[comp_name, irhs][-hist_len:]
                relv_hist_vars = temp_hist_vars[comp_name, irhs][-hist_len:]

                t_start = latest_state_substep / self.nsubsteps
                t_end = isubstep / self.nsubsteps

                if not self.static_dt:
                    time_hist_var = var(name_gen("time_hist"))
                    cb(time_hist_var, array(hist_len))

                    for ii in range(hist_len):
                        cb(time_hist_var[ii], relv_time_hist[ii] - self.t)

                    time_hist = time_hist_var
                    t_start *= self.dt
                    t_end *= self.dt
                    dt_factor = 1

                else:
                    time_hist = relv_time_hist
                    dt_factor = self.dt

                from leap.multistep import (
                        AdamsMonomialIntegrationFunctionFamily,
                        emit_adams_integration,
                        emit_adams_extrapolation)

                if self.is_ode_component[comp_name]:
                    contrib = dt_factor*emit_adams_integration(
                                cb, name_gen,
                                AdamsMonomialIntegrationFunctionFamily(rhs.order),
                                time_hist, relv_hist_vars,
                                t_start, t_end)

                else:
                    contrib = emit_adams_extrapolation(
                                cb, name_gen,
                                AdamsMonomialIntegrationFunctionFamily(rhs.order),
                                time_hist, relv_hist_vars,
                                t_end)

                contribs.append(contrib)
                contrib_explanations.append(
                        self.StateContribExplanation(
                            rhs=rhs.func_name,
                            from_substeps=relv_hist_substeps,
                            using=relv_hist_vars))

            state_var = var(
                    name_gen(
                        "state_{comp_name}_sub{isubstep}"
                        .format(comp_name=comp_name, isubstep=isubstep)))

            if self.is_ode_component[comp_name]:
                state_expr = latest_state + sum(contribs)
            else:
                state_expr = sum(contribs)

            if comp_name in self.state_filters:
                state_expr = self.state_filters[comp_name](state_expr)
            cb(state_var, state_expr)

            # Only keep temporary state if integrates exactly
            # one interval ahead for the fastest right-hand side,
            # which is the expected rate.
            #
            # - If it integrates further, it's a poor-quality
            #   extrapolation that should probably not be reused.
            #
            # - If it integrates less far, then by definition it is
            #   not used for any state updates, and we don't gain
            #   anything by keeping the temporary around, since the
            #   same extrapolation can be recomputed.

            keep_temp_state = (
                    isubstep - latest_state_substep == min(
                        rhs.interval for rhs in rhss))
            if keep_temp_state:
                states.append((isubstep, state_var))

            if self.is_ode_component[comp_name]:
                explainer.integrate_to(comp_name, state_var.name,
                        latest_state_substep, isubstep, latest_state,
                        contrib_explanations)
            else:
                explainer.extrapolate_to(comp_name, state_var.name,
                        latest_state_substep, isubstep, latest_state,
                        contrib_explanations)

            return state_var

        # }}}

        # {{{ update_hist

        def update_hist(comp_idx, irhs, isubstep):
            comp_name = self.component_names[comp_idx]

            rhs = self.rhss[comp_idx][irhs]

            # {{{ get arguments together

            progress_frac = isubstep / self.nsubsteps
            t_expr = self.t + self.dt * progress_frac

            kwargs = {
                    self.comp_name_to_kwarg_name[arg_comp_name]:
                    get_state(arg_comp_name, isubstep)
                    for arg_comp_name in rhs.arguments}

            # }}}

            rhs_var = var(
                    name_gen(
                        "rhs_{comp_name}_rhs{irhs}_sub{isubstep}"
                        .format(comp_name=comp_name, irhs=irhs, isubstep=isubstep)))

            cb(rhs_var, var(rhs.func_name)(t=t_expr, **kwargs))

            temp_hist_substeps[comp_name, irhs].append(isubstep)

            if not self.static_dt:
                t_var = var(
                        name_gen(
                            "t_{comp_name}_rhs{irhs}_sub{isubstep}"
                            .format(
                                comp_name=comp_name,
                                irhs=irhs,
                                isubstep=isubstep)))
                cb(t_var, t_expr)
                temp_time_vars[comp_name, irhs].append(t_var)

            else:
                temp_time_vars[comp_name, irhs].append(progress_frac)

            temp_hist_vars[comp_name, irhs].append(rhs_var)

            explainer.eval_rhs(
                    rhs_var.name, comp_name, rhs.func_name, isubstep, kwargs)

            # {{{ invalidate computed states, if requested

            if rhs.invalidate_computed_state:
                for other_comp_name, other_component_rhss in zip(
                        self.component_names, self.rhss):
                    do_invalidate = False
                    for _other_rhs in enumerate(other_component_rhss):
                        if comp_name in rhs.arguments:
                            do_invalidate = True
                            break

                    if do_invalidate:
                        computed_states[other_comp_name][:] = [
                                (istate_substep, state)

                                for istate_substep, state in
                                computed_states[other_comp_name]

                                # Only earlier states live.
                                if istate_substep < isubstep
                                ]

            # }}}

        # }}}

        def norm(expr):
            return var("<builtin>norm_2")(expr)

        def check_history_consistency():
            # At the start of a macrostep, ensure that the last computed
            # RHS history corresponds to the current state
            for comp_name, component_rhss in zip(self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    t_expr = self.t
                    kwargs = {
                            self.comp_name_to_kwarg_name[arg_comp_name]:
                            get_state(arg_comp_name, 0)
                            for arg_comp_name in rhs.arguments}
                    test_rhs_var = var(
                            name_gen(
                                "test_rhs_{comp_name}_rhs{irhs}_0"
                                .format(comp_name=comp_name, irhs=irhs)))

                    cb(test_rhs_var, var(rhs.func_name)(t=t_expr, **kwargs))
                    # Compare this computed RHS with the 0th history point using
                    # built-in norm.

                    zeroth_hist = temp_hist_vars[comp_name, irhs][-1]
                    rel_rhs_error = (
                            norm(test_rhs_var - zeroth_hist)
                            /  # noqa: W504
                            norm(test_rhs_var))

                    cb("rel_rhs_error", rel_rhs_error)

                    # cb((), "<builtin>print(rel_rhs_error)")

                    if rhs.rhs_policy == rhs_policy.early:
                        # Check for scheme-order accuracy
                        if self.early_hist_consistency_threshold is not None:
                            with cb.if_("rel_rhs_error", ">=",
                                    self.early_hist_consistency_threshold):
                                cb((), "<builtin>print(rel_rhs_error)")
                                cb.raise_(InconsistentHistoryError,
                                        "MRAdams: top-of-history for RHS "
                                        f"'{rhs.func_name}' is not "
                                        "consistent with current state")
                        else:
                            cb.raise_(InconsistentHistoryError,
                                    "MRAdams: RHS '{rhs.func_name}' has early "
                                    "policy and requires relaxed threshold input")
                    else:
                        # Check for floating-point accuracy
                        with cb.if_("rel_rhs_error", ">=",
                                self.hist_consistency_threshold):
                            cb.raise_(InconsistentHistoryError,
                                    "MRAdams: top-of-history for RHS "
                                    f"'{rhs.func_name}' is not "
                                    "consistent with current state")

        # {{{ run_substep_loop

        def run_substep_loop():
            # Check last history value from previous macrostep
            if self.hist_consistency_threshold is not None:
                check_history_consistency()

            for isubstep in range(self.nsubsteps+1):
                for comp_idx, (comp_name, component_rhss) in enumerate(
                        zip(self.component_names, self.rhss)):
                    for irhs, rhs in enumerate(component_rhss):
                        if isubstep % rhs.interval != 0:
                            continue

                        if isubstep > 0:
                            # {{{ finish up prior step

                            if rhs.rhs_policy == rhs_policy.early_and_late:
                                temp_hist_substeps[comp_name, irhs].pop()
                                temp_time_vars[comp_name, irhs].pop()
                                temp_hist_vars[comp_name, irhs].pop()
                                explainer.roll_back_history(rhs.func_name)

                            if rhs.rhs_policy in [
                                    rhs_policy.early_and_late, rhs_policy.late]:
                                update_hist(comp_idx, irhs, isubstep)

                            # }}}

                        if isubstep < self.nsubsteps:
                            # {{{ start up a new substep

                            if rhs.rhs_policy in [
                                    rhs_policy.early, rhs_policy.early_and_late]:
                                update_hist(comp_idx, irhs, isubstep + rhs.interval)

                            # }}}

        run_substep_loop()

        # }}}

        log_hist_state()

        end_states = [
            get_state(component_name, self.nsubsteps)
            for component_name in self.component_names]

        # {{{ commit temp history to permanent history

        def commit_temp_hist_vars():
            for comp_name, component_rhss in zip(self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    key = comp_name, irhs

                    if not self.static_dt:
                        for time_var, time_expr in zip(
                                self.time_vars[key],
                                temp_time_vars[comp_name,
                                    irhs][-rhs.history_length:]):
                            cb(time_var, time_expr)

                    for hist_var, hist_expr in zip(
                            self.history_vars[key],
                            temp_hist_vars[comp_name, irhs][-rhs.history_length:]):
                        cb(hist_var, hist_expr)

        commit_temp_hist_vars()

        # }}}

        # TODO: Figure out more spots to yield intermediate state
        for component_name, state in zip(self.component_names, end_states):
            if self.is_ode_component[component_name]:
                cb.yield_state(
                        state,
                        component_name, self.t + self.dt, "final")

            cb(var("<state>"+component_name), state)

        cb(self.t, self.t + self.dt)

    # }}}

    # {{{ generation entrypoint

    def generate(self, explainer=None):
        """
        :arg explainer: a subclass of :class:`SchemeExplainerBase`, possibly
            :class:`TextualSchemeExplainer`, or *None*.
        :returns: :class:`dagrt.language.DAGCode`
        """
        if explainer is None:
            explainer = SchemeExplainerBase()

        from dagrt.language import DAGCode, CodeBuilder

        with CodeBuilder(name="initialization") as cb_init:
            self.emit_initialization(cb_init)

        with CodeBuilder(name="primary") as cb_primary:
            self.emit_adams_method(cb_primary, explainer)

        with CodeBuilder(name="bootstrap") as cb_bootstrap:
            self.emit_rk_bootstrap(cb_bootstrap)

        return DAGCode(
                phases={
                    "initialization": cb_init.as_execution_phase("bootstrap"),
                    "bootstrap": cb_bootstrap.as_execution_phase("bootstrap"),
                    "primary": cb_primary.as_execution_phase("primary"),
                    },
                initial_phase="initialization")

        # }}}

# }}}


# {{{ two-rate compatibility shim

class TwoRateAdamsBashforthMethodBuilder(MultiRateMultiStepMethodBuilder):
    methods = [
            "Sqrs",
            "Sqr",
            "Sqs",
            "Sq",

            "Srsf",
            "Srs",
            "Srf",
            "Sr",

            "Ssf",
            "Ss",
            "Sf",
            "S",

            "Fqsr",
            "Fqs",
            "Fq",

            "Fsfr",
            "Fsf",
            "Fsr",
            "Fs",

            "Ffr",
            "Ff",
            "F"
            ]

    def __init__(self, method, order, step_ratio,
            slow_state_filter_name=None,
            fast_state_filter_name=None,
            static_dt=False, hist_consistency_threshold=None,
            early_hist_consistency_threshold=None,
            hist_length_slow=None,
            hist_length_fast=None):
        from warnings import warn
        warn("TwoRateAdamsBashforthMethodBuilder "
                "is a compatibility shim that should no "
                "longer be used. Use the fully general "
                "MultiRateMultiStepMethodBuilder interface instead.",
                DeprecationWarning, stacklevel=2)

        if "S" in method:
            s2s_policy = rhs_policy.early
        elif "F" in method:
            s2s_policy = rhs_policy.late
        else:
            raise ValueError("expecting 'F' or 'S' in method")

        if "r" in method:
            s2s_policy = rhs_policy.early_and_late

        if "q" in method:
            s2f_interval = 1
        else:
            s2f_interval = step_ratio

        if "s" in method:
            f2s_policy = rhs_policy.early
        else:
            f2s_policy = rhs_policy.late

        if "f" in method:
            s2f_policy = rhs_policy.early
        else:
            s2f_policy = rhs_policy.late

        if hist_length_slow is None:
            hist_length_slow = order

        if hist_length_fast is None:
            hist_length_slow = order

        super().__init__(
                order,
                (
                    (
                        "dt", "fast", "=",
                        MultiRateHistory(1, "<func>f2f", ("fast", "slow",),
                            hist_length=hist_length_fast),
                        MultiRateHistory(s2f_interval, "<func>s2f",
                            ("fast", "slow",), rhs_policy=s2f_policy,
                            hist_length=hist_length_fast),
                        ),
                    (
                        "dt", "slow", "=",
                        MultiRateHistory(step_ratio, "<func>f2s", ("fast", "slow",),
                            rhs_policy=f2s_policy, hist_length=hist_length_slow),
                        MultiRateHistory(step_ratio, "<func>s2s", ("fast", "slow",),
                            rhs_policy=s2s_policy, hist_length=hist_length_slow),
                        ),),

                state_filter_names={
                    "fast": fast_state_filter_name,
                    "slow": slow_state_filter_name,
                    },

                # This is a hack to avoid having to change the 2RAB test
                # cases, which use these arguments
                component_arg_names=("f", "s"),

                static_dt=static_dt,
                hist_consistency_threshold=hist_consistency_threshold,
                early_hist_consistency_threshold=early_hist_consistency_threshold)

# }}}


# {{{ scheme explainers

class SchemeExplainerBase:

    def log_hist_state(self, hist_substeps):
        pass

    def integrate_to(self, component_name, var_name,
            from_substep, to_substep, latest_state,
            contrib_explanations):
        pass

    def extrapolate_to(self, component_name, var_name,
            base_substep, to_substep, latest_state, contrib_explanations):
        pass

    def eval_rhs(self, rhs_var, comp_name, rhs_name, isubstep, kwargs):
        pass

    def roll_back_history(self, rhs_name):
        pass


class TextualSchemeExplainer(SchemeExplainerBase):
    """
    .. automethod:: __init__
    .. automethod:: __str__
    """
    def __init__(self):
        self.lines = []

    def __str__(self):
        return "\n".join(self.lines)

    def log_hist_state(self, hist_substeps):
        self.lines.append("HISTORY:")
        for rhs_name, rhs_hist_substeps_and_vars in hist_substeps.items():
            self.lines.append(
                    "    {rhs}: {substeps}"
                    .format(
                        rhs=rhs_name.replace("<func>", ""),
                        substeps=", ".join(
                            str(i)+":"+var
                            for i, var in zip(*rhs_hist_substeps_and_vars))))

    def _write_contrib_explanations(self, contrib_explanations):
        for contrib in contrib_explanations:
            self.lines.append(
                    "    {rhs}: {states}"
                    .format(
                        rhs=contrib.rhs.replace("<func>", ""),
                        states=" ".join(
                            f"{substep}:{name}"
                            for substep, name in zip(
                                contrib.from_substeps, contrib.using))))

    def integrate_to(self, component_name, var_name,
            from_substep, to_substep, latest_state,
            contrib_explanations):
        self.lines.append(
                "{verb}: {var_name} <- "
                "FROM {from_substep} ({latest_state}) TO {to_substep}:"
                .format(
                    verb=("INTEGRATE EXTRAPOLANT" if from_substep < to_substep
                        else "INTEGRATE INTERPOLANT"),
                    var_name=var_name,
                    from_substep=from_substep,
                    to_substep=to_substep,
                    latest_state=latest_state,
                    ))

        self._write_contrib_explanations(contrib_explanations)

    def extrapolate_to(self, component_name, var_name,
            base_substep, to_substep, latest_state,
            contrib_explanations):
        self.lines.append(
                "{verb}: {var_name} <- "
                "FROM {base_substep} ({latest_state}) TO {to_substep}:"
                .format(
                    verb=("EXTRAPOLATE" if base_substep < to_substep
                        else "INTERPOLATE"),
                    var_name=var_name,
                    base_substep=base_substep,
                    to_substep=to_substep,
                    latest_state=latest_state,
                    ))

        self._write_contrib_explanations(contrib_explanations)

    def eval_rhs(self, rhs_var, comp_name, rhs_name, isubstep, kwargs):
        self.lines.append(
                "EVAL {rhs_var} <- {rhs_name}(t={isubstep}, {kwargs})"
                .format(
                    rhs_var=rhs_var,
                    rhs_name=rhs_name.replace("<func>", ""),
                    isubstep=isubstep,
                    kwargs=", ".join(
                        f"{k}={v}"
                        for k, v in sorted(kwargs.items()))))

    def roll_back_history(self, rhs_name):
        self.lines.append(f"ROLL BACK {rhs_name}")

# }}}


# {{{ utils

def _gcd(a, b):
    while a:
        b, a = a, b % a
    return b


def gcd(args):
    args = iter(args)
    result = next(args)
    for arg in args:
        result = _gcd(result, arg)
    return result


def lcm(args):
    args = iter(args)
    result = next(args)
    for arg in args:
        result = (result * arg) // _gcd(result, arg)
    return result

# }}}

# vim: foldmethod=marker
