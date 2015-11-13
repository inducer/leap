"""Multirate-AB ODE method."""

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

from pytools import Record
from leap import Method
import six.moves
from pymbolic import var


__doc__ = """
Multi-rate time integration
===========================

.. autoclass:: rhs_policy
.. autoclass:: RHS
.. autoclass:: MultiRateMultiStepMethod

Scheme explanation
------------------

.. autoclass:: SchemeExplainerBase
.. autoclass:: TextualSchemeExplainer
.. autoclass:: TeXDiagramSchemeExplainer
"""


# {{{ utilities

def _linear_comb(coefficients, vectors):
    from operator import add
    return six.moves.reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))

# }}}


# {{{ system description

class rhs_policy:
    late = 0
    early = 1
    early_and_late = 2


class RHS(Record):
    def __init__(self, interval, func_name, arguments=None, order=None,
            rhs_policy=rhs_policy.late, invalidate_computed_state=False):
        """
        :arg interval: An integer indicating the interval (relative to the
            smallest available timestep) at which this right-hand side
            function is to be called.
        :arg arguments: A tuple of component names
            (see :class:`MultiRateMultiStepMethod`)
            which are passed to this right-hand side function.
        :arg order: The AB approximation order to be used for this RHS
            history, or None if the method default is to be used.
        :arg rhs_policy: One of the constants in :class:`rhs_policy`
        :arg invalidate_dependent_state: Whether evaluating this
            right-hand side should force a recomputation of any
            state that depended upon now-superseded state.
        """
        super(RHS, self).__init__(
                interval=interval,
                func_name=func_name,
                arguments=arguments,
                order=order,
                rhs_policy=rhs_policy,
                invalidate_computed_state=invalidate_computed_state)

    @property
    def history_length(self):
        return self.order

# }}}


# {{{ method

class MultiRateMultiStepMethod(Method):
    """Simultaneously timesteps multiple parts of an ODE system,
    each with adjustable orders, rates, and dependencies.

    [1] C.W. Gear and D.R. Wells, "Multirate linear multistep methods," BIT
    Numerical Mathematics,  vol. 24, Dec. 1984,pg. 484-502.
    """

    # {{{ constructor

    def __init__(self, default_order, component_names,
            rhss,
            state_filter_names=None,
            component_arg_names=None):
        """
        :arg default_order: The order to be used for right-hand sides
            where no differing order is specified.
        :arg component_names: A tuple of names of the components
            of the ODE system to be integrated.
        :arg rhss: A tuple of the same length as *component_names*,
            where each entry in the tuple is a further tuple
            of :class:`RHS` instances indicating the right-hand-sides
            contributing to this component.
        :arg state_filter_names: *None* or a tuple of state filter names
            (or *None* values) of the same length as *component_names*.
        :arg component_arg_names: A tuple of names of the components
            to be used as keywords for passing arguments to the right
            hand sides in *rhss*.
        """
        super(MultiRateMultiStepMethod, self).__init__()

        # Variables
        from pymbolic import var

        self.t = var('<t>')
        self.dt = var('<dt>')
        self.bootstrap_step = var('<p>bootstrap_step')

        if len(rhss) != len(component_names):
            raise ValueError("rhss and component_names must have the same length")

        if state_filter_names is None:
            state_filter_names = (None,) * len(component_names)

        if len(state_filter_names) != len(component_names):
            raise ValueError("rhss and component_names must have the same length")

        self.state_filters = tuple(
                var("<func>" + state_filter_name)
                if state_filter_name is not None
                else None
                for state_filter_name in state_filter_names)

        # {{{ prepropcess rhss

        new_rhss = []
        for component_rhss in rhss:
            new_component_rhss = []
            for rhs in component_rhss:
                order = rhs.order
                if order is None:
                    order = default_order

                arguments = rhs.arguments
                if arguments is None:
                    arguments = self.arguments

                new_component_rhss.append(
                        rhs.copy(
                            order=order,
                            arguments=arguments))

            new_rhss.append(tuple(new_component_rhss))

        self.rhss = new_rhss
        del new_rhss
        del rhss

        # }}}

        self.component_names = component_names

        if component_arg_names is None:
            component_arg_names = component_names

        self.comp_name_to_kwarg_name = dict(
                zip(component_names, component_arg_names))

        self.max_order = max(rhs.order
                for component_rhss in self.rhss
                for rhs in component_rhss)

        # {{{ process intervals

        intervals = sorted(rhs.interval
                for component_rhss in self.rhss
                for rhs in component_rhss)

        substep_counts = []
        for i in range(1, len(intervals)):
            last_interval = intervals[i-1]
            interval = intervals[i]

            if interval % last_interval != 0:
                raise ValueError(
                        "intervals are not integer multiples of each other: "
                        + ", ".join(str(intv) for intv in intervals))

            substep_counts.append(interval // last_interval)

        if min(intervals) != 1:
            raise ValueError("the smallest interval is not 1")

        self.intervals = intervals
        self.substep_counts = substep_counts

        # }}}

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
                        '<p>t_%s_rhs%d_hist_%d_ago' % (comp_name, irhs, past)))
                    hist_vars.insert(0, var(
                        '<p>hist_%s_rhs%d_hist_%d_ago' % (comp_name, irhs, past)))

                self.time_vars[key] = t_vars
                self.history_vars[key] = hist_vars

        self.state_vars = tuple(
                var("<state>" + comp_name) for comp_name in self.component_names)

    # }}}

    @property
    def nsubsteps(self):
        return max(self.intervals)

    def emit_initialization(self, cb):
        """Initialize method variables."""

        cb(self.bootstrap_step, 0)

    # {{{ rk bootstrap: step

    def emit_small_rk_step(self, cb, name_prefix, name_gen, entry_rhss):
        """Emit a single step of an RK method."""

        from leap.rk import ORDER_TO_RK_METHOD
        rk_method = ORDER_TO_RK_METHOD[self.max_order]
        rk_tableau = tuple(zip(rk_method.c, rk_method.a_explicit))
        rk_coeffs = rk_method.output_coeffs

        def make_stage_history(prefix):
            return [var(prefix + "_stage" + str(i)) for i in range(len(rk_tableau))]

        stage_rhss = {}
        for comp_name, component_rhss in zip(self.component_names, self.rhss):
            for irhs, rhs in enumerate(component_rhss):
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
                    for irhs, rhs in enumerate(component_rhss):
                        cb(stage_rhss[comp_name, irhs][istage],
                                entry_rhss[comp_name, irhs])

            else:
                component_state_ests = {}

                for icomp, (comp_name, component_rhss) in enumerate(
                        zip(self.component_names, self.rhss)):

                    contribs = []
                    for irhs, rhs in enumerate(component_rhss):
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
                    if self.state_filters[icomp] is not None:
                        state_expr = self.state_filters[icomp](state_expr)

                    cb(state_var, state_expr)

                    component_state_ests[comp_name] = state_var

                for comp_name, component_rhss in zip(
                        self.component_names, self.rhss):
                    for irhs, rhs in enumerate(component_rhss):
                        kwargs = dict(
                                (self.comp_name_to_kwarg_name[arg_comp_name],
                                    component_state_ests[arg_comp_name])
                                for arg_comp_name in rhs.arguments)
                        cb(stage_rhss[comp_name, irhs][istage],
                                var(rhs.func_name)(
                                    t=self.t + (c/self.nsubsteps) * self.dt,
                                    **kwargs))

        cb.fence()

        component_state_ests = {}

        for icomp, (comp_name, component_rhss) in enumerate(
                zip(self.component_names, self.rhss)):

            contribs = []
            for irhs, rhs in enumerate(component_rhss):
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
            if self.state_filters[icomp] is not None:
                state_expr = self.state_filters[icomp](state_expr)

            cb(state_var, state_expr)

            component_state_ests[comp_name] = state_var

        cb.fence()

        for component_name in self.component_names:
            state = component_state_ests[component_name]
            cb.yield_state(
                    state,
                    component_name, self.t + self.dt/self.nsubsteps,
                    "bootstrap")

            cb(var("<state>"+component_name), state)

        cb.fence()

        cb(self.t, self.t + self.dt/self.nsubsteps)

    # }}}

    # {{{ rk bootstrap: overall control

    def emit_rk_bootstrap(self, cb):
        """Initialize the stepper with an RK method. Return the code that
        computes the startup history."""

        bootstrap_steps = self.max_order - 1

        final_iglobal_substep = bootstrap_steps * self.nsubsteps

        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        for isubstep in range(self.nsubsteps + 1):
            name_prefix = 'substep' + str(isubstep)

            # {{{ compute current_rhss

            current_rhss = {}

            for comp_name, component_rhss in zip(
                    self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    rhs_var = var(
                        name_gen(
                            "{name_prefix}_start_{comp_name}_rhs{irhs}"
                            .format(name_prefix=name_prefix, comp_name=comp_name,
                                irhs=irhs)))

                    kwargs = dict(
                            (self.comp_name_to_kwarg_name[arg_comp_name],
                                var("<state>" + arg_comp_name))
                            for arg_comp_name in rhs.arguments)

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
                                    < rhs.order)):

                            intervals_from_start = (
                                    substeps_from_start // rhs.interval)

                            i = rhs.order - 1 - intervals_from_start
                            assert i >= 0

                            with cb.if_(self.bootstrap_step, "==", test_step):
                                cb(self.time_vars[comp_name, irhs][i], self.t)
                                cb(self.history_vars[comp_name, irhs][i],
                                        current_rhss[comp_name, irhs])

            # }}}

            if isubstep == self.nsubsteps:
                cb.fence()
                cb(self.bootstrap_step, self.bootstrap_step + 1)
                break

            if isubstep == 0:
                with cb.if_(self.bootstrap_step, "==", bootstrap_steps):
                    cb.state_transition("primary")

            cb.fence()

            self.emit_small_rk_step(cb, name_prefix, name_gen, current_rhss)

        return cb

    # }}}

    class StateContribExplanation(Record):
        pass

    # {{{ main method generation

    def emit_ab_method(self, cb, explainer):
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

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
                        -rhs.interval*(rhs.order-1), 1, rhs.interval))
                    temp_time_vars[key] = self.time_vars[key][:]
                    temp_hist_vars[key] = self.history_vars[key][:]

        fill_temp_hist_vars()

        # }}}

        def log_hist_state():
            explainer.log_hist_state(dict(
                (rhs.func_name, (
                    temp_hist_substeps[comp_name, irhs][-rhs.history_length::],
                    [v.name
                        for v in
                        temp_hist_vars[comp_name, irhs][-rhs.history_length::]]))
                for comp_name, component_rhss in zip(self.component_names, self.rhss)
                for irhs, rhs in enumerate(component_rhss)))

        log_hist_state()

        # A mapping from component_name to a list of tuples
        # (substep_level, state_var). This mapping is ordered
        # by substep_level.
        computed_states = dict(
                (comp_name, [
                    (0, state_var)
                    ])
                for comp_name, state_var in zip(
                    self.component_names, self.state_vars))

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

            array = var("<builtin>array")
            linear_solve = var("<builtin>linear_solve")

            contribs = []
            contrib_explanations = []

            for irhs, rhs in enumerate(rhss):
                cb.fence()
                hist_len = rhs.history_length
                order = rhs.order

                relv_hist_substeps = temp_hist_substeps[comp_name, irhs][-hist_len:]
                relv_time_hist = temp_time_vars[comp_name, irhs][-hist_len:]
                relv_hist_vars = temp_hist_vars[comp_name, irhs][-hist_len:]

                # {{{ compute AB coefficients

                # use:
                # Vandermonde^T * ab_coeffs = integrate(t_start, t_end, monomials)

                vdmt = var(name_gen("vdm_transpose"))
                cb(vdmt, array(order*hist_len))

                coeff_rhs = var(name_gen("coeff_rhs"))
                cb(coeff_rhs, array(hist_len))

                i = var(name_gen("vdm_i"))
                j = var(name_gen("vdm_j"))

                time_hist_var = var(name_gen("time_hist"))
                cb(time_hist_var, array(hist_len))

                for ii in range(hist_len):
                    cb(time_hist_var[ii], relv_time_hist[ii] - self.t)

                cb(vdmt[i + j*order], time_hist_var[j]**i,
                    loops=[(i.name, 0, order), (j.name, 0, hist_len)])

                t_start = self.dt * latest_state_substep / self.nsubsteps
                t_end = self.dt * isubstep / self.nsubsteps

                cb(coeff_rhs[i],
                        1/(i+1) * (t_end**(i+1) - t_start**(i+1)),
                        loops=[(i.name, 0, order)])

                ab_coeffs = var(name_gen("ab_coeffs"))
                cb(ab_coeffs, linear_solve(vdmt, coeff_rhs, order, 1))

                # }}}

                state_contrib_var = var(
                        name_gen(
                            "state_contrib_{comp_name}_rhs{irhs}"
                            .format(comp_name=comp_name, irhs=irhs)))

                cb(state_contrib_var,
                        _linear_comb(
                            [ab_coeffs[ii] for ii in range(hist_len)],
                            relv_hist_vars))

                contribs.append(state_contrib_var)
                contrib_explanations.append(
                        self.StateContribExplanation(
                            rhs=rhs.func_name,
                            from_substeps=relv_hist_substeps,
                            using=relv_hist_vars))

                cb.fence()

            state_var = var(
                    name_gen(
                        "state_{comp_name}_sub{isubstep}"
                        .format(comp_name=comp_name, isubstep=isubstep)))

            state_expr = latest_state + sum(contribs)
            if self.state_filters[comp_index] is not None:
                state_expr = self.state_filters[comp_index](state_expr)
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

            explainer.integrate_to(comp_name, state_var.name,
                    latest_state_substep, isubstep, latest_state,
                    contrib_explanations)

            return state_var

        # }}}

        # {{{ update_hist

        def update_hist(comp_idx, irhs, isubstep):
            comp_name = self.component_names[comp_idx]

            rhs = self.rhss[comp_idx][irhs]

            # {{{ get arguments together

            t_var = var(
                    name_gen(
                        "t_{comp_name}_rhs{irhs}_sub{isubstep}"
                        .format(comp_name=comp_name, irhs=irhs, isubstep=isubstep)))
            t_expr = self.t + self.dt * isubstep / self.nsubsteps
            cb(t_var, t_expr)

            kwargs = dict(
                    (self.comp_name_to_kwarg_name[arg_comp_name],
                        get_state(arg_comp_name, isubstep))
                    for arg_comp_name in rhs.arguments)

            # }}}

            rhs_var = var(
                    name_gen(
                        "rhs_{comp_name}_rhs{irhs}_sub{isubstep}"
                        .format(comp_name=comp_name, irhs=irhs, isubstep=isubstep)))

            cb(rhs_var, var(rhs.func_name)(t=t_expr, **kwargs))

            temp_hist_substeps[comp_name, irhs].append(isubstep)
            temp_time_vars[comp_name, irhs].append(t_var)
            temp_hist_vars[comp_name, irhs].append(rhs_var)

            explainer.eval_rhs(
                    rhs_var.name, comp_name, rhs.func_name, isubstep, kwargs)

            # {{{ invalidate computed states, if requested

            if rhs.invalidate_computed_state:
                for other_comp_name, other_component_rhss in zip(
                        self.component_names, self.rhss):
                    do_invalidate = False
                    for other_rhs in enumerate(other_component_rhss):
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

        # {{{ run_substep_loop

        def run_substep_loop():
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

        cb.fence()

        log_hist_state()

        end_states = [
            get_state(component_name, self.nsubsteps)
            for component_name in self.component_names]

        cb.fence()

        # {{{ commit temp history to permanent history

        def commit_temp_hist_vars():
            for comp_name, component_rhss in zip(self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    key = comp_name, irhs

                    for time_var, time_expr in zip(
                            self.time_vars[key],
                            temp_time_vars[comp_name, irhs][-rhs.order:]):
                        cb(time_var, time_expr)
                        cb.fence()

                    for hist_var, hist_expr in zip(
                            self.history_vars[key],
                            temp_hist_vars[comp_name, irhs][-rhs.order:]):
                        cb(hist_var, hist_expr)
                        cb.fence()

        commit_temp_hist_vars()

        # }}}

        # TODO: Figure out more spots to yield intermediate state
        for component_name, state in zip(self.component_names, end_states):
            cb.yield_state(
                    state,
                    component_name, self.t + self.dt, "final")

            cb(var("<state>"+component_name), state)

        cb.fence()

        cb(self.t, self.t + self.dt)

    # }}}

    # {{{ generation entrypoint

    def generate(self, explainer=None):
        if explainer is None:
            explainer = SchemeExplainerBase()

        from dagrt.language import DAGCode, CodeBuilder

        with CodeBuilder(label="initialization") as cb_init:
            self.emit_initialization(cb_init)

        with CodeBuilder(label="primary") as cb_primary:
            self.emit_ab_method(cb_primary, explainer)

        with CodeBuilder(label="bootstrap") as cb_bootstrap:
            self.emit_rk_bootstrap(cb_bootstrap)

        states = {}
        states["initialization"] = cb_init.as_execution_state("bootstrap")
        states["bootstrap"] = cb_bootstrap.as_execution_state("bootstrap")
        states["primary"] = cb_primary.as_execution_state("primary")

        return DAGCode(
            instructions=cb_init.instructions | cb_bootstrap.instructions |
            cb_primary.instructions,
            states=states,
            initial_state="initialization")

        # }}}

# }}}


# {{{ two-rate compatibility shim

class TwoRateAdamsBashforthMethod(MultiRateMultiStepMethod):
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
            fast_state_filter_name=None):
        from warnings import warn
        warn("TwoRateAdamsBashforthMethod is a compatibility shim that should no "
                "longer be used. Use the fully general "
                "MultiRateMultiStepMethod interface instead.",
                DeprecationWarning, stacklevel=2)

        if "S" in method:
            s2s_policy = rhs_policy.early
        else:
            s2s_policy = rhs_policy.late

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

        super(TwoRateAdamsBashforthMethod, self).__init__(
                order,
                component_names=("fast", "slow",),
                rhss=(
                    (
                        RHS(1, "<func>f2f", ("fast", "slow",)),
                        RHS(s2f_interval, "<func>s2f", ("fast", "slow",),
                            rhs_policy=s2f_policy),
                        ),
                    (
                        RHS(step_ratio, "<func>f2s", ("fast", "slow",),
                            rhs_policy=f2s_policy),
                        RHS(step_ratio, "<func>s2s", ("fast", "slow",),
                            rhs_policy=s2s_policy),
                        ),),

                state_filter_names=(fast_state_filter_name, slow_state_filter_name),

                # This is a hack to avoid having to change the 2RAB test
                # cases, which use these arguments
                component_arg_names=("f", "s"))

# }}}


# {{{ scheme explainers

class SchemeExplainerBase(object):
    """
    .. automethod:: evaluate_rhs
    .. automethod:: integrate
    """

    def log_hist_state(self, hist_substeps):
        pass

    def integrate_to(self, component_name, var_name,
            from_substep, to_substep, latest_state,
            contrib_explanations):
        pass

    def eval_rhs(self, rhs_var, comp_name, rhs_name, isubstep, kwargs):
        pass

    def roll_back_history(self, rhs_name):
        pass


class TextualSchemeExplainer(SchemeExplainerBase):
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

    def integrate_to(self, component_name, var_name,
            from_substep, to_substep, latest_state,
            contrib_explanations):
        self.lines.append(
                "INTEGRATE: {var_name} <- "
                "FROM {from_substep} ({latest_state}) TO {to_substep}:"
                .format(
                    var_name=var_name,
                    from_substep=from_substep,
                    to_substep=to_substep,
                    latest_state=latest_state,
                    ))

        for contrib in contrib_explanations:
            self.lines.append(
                    "    {rhs}: {states}"
                    .format(
                        rhs=contrib.rhs.replace("<func>", ""),
                        states=" ".join(
                            "%d:%s" % (substep, name)
                            for substep, name in zip(
                                contrib.from_substeps, contrib.using))))

    def eval_rhs(self, rhs_var, comp_name, rhs_name, isubstep, kwargs):
        self.lines.append(
                "EVAL {rhs_var} <- {rhs_name}(t={isubstep}, {kwargs})"
                .format(
                    rhs_var=rhs_var,
                    comp_name=comp_name,
                    rhs_name=rhs_name.replace("<func>", ""),
                    isubstep=isubstep,
                    kwargs=", ".join(
                        "%s=%s" % (k, v)
                        for k, v in sorted(kwargs.items()))))

    def roll_back_history(self, rhs_name):
        self.lines.append("ROLL BACK %s" % rhs_name)


class TeXDiagramSchemeExplainer(SchemeExplainerBase):
    def __init__(self):
        self.lines = []

    def __str__(self):
        return "\n".join(self.lines)


# }}}


# vim: foldmethod=marker
