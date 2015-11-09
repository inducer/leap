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
.. autoclass:: rhs_mode
.. autoclass:: RHS
.. autoclass:: MultiRateAdamsBashforthMethod
"""


# {{{ utilities

def _linear_comb(coefficients, vectors):
    from operator import add
    return six.moves.reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))

# }}}


# {{{ system description

class rhs_mode:
    late = 0
    early = 1
    early_and_late = 2


class RHS(Record):
    def __init__(self, interval, func_name, arguments=None, order=None,
            rhs_mode=rhs_mode.late, invalidate_computed_state=False):
        """
        :arg interval: An integer indicating the interval (relative to the
            smallest available timestep) at which this right-hand side
            function is to be called.
        :arg arguments: A tuple of component names
            (see :class:`MultiRateAdamsBashforthMethod`)
            which are passed to this right-hand side function.
        :arg order: The AB approximation order to be used for this RHS
            history, or None if the method default is to be used.
        :arg rhs_mode: One of the constants in :class:`rhs_mode`
        :arg invalidate_dependent_state: Whether evaluating this
            right-hand side should force a recomputation of any
            state that depended upon now-superseded state.
        """
        super(RHS, self).__init__(
                interval=interval,
                func_name=func_name,
                arguments=arguments,
                order=order,
                rhs_mode=rhs_mode,
                invalidate_computed_state=invalidate_computed_state)

# }}}


# {{{ method

class MultiRateAdamsBashforthMethod(Method):
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
        super(MultiRateAdamsBashforthMethod, self).__init__()

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

                t_vars = []
                hist_vars = []
                for past in range(rhs.order):
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

    # {{{ main method generation

    def emit_ab_method(self, cb):
        from pytools import UniqueNameGenerator
        name_gen = UniqueNameGenerator()

        # {{{ make temporary copies of time/hist_vars for "early" rhs_mode

        temp_time_vars = {}
        temp_hist_vars = {}

        def fill_temp_hist_vars():
            for comp_name, component_rhss in zip(self.component_names, self.rhss):
                for irhs, rhs in enumerate(component_rhss):
                    key = comp_name, irhs

                    temp_time_vars[key] = self.time_vars[key][:]
                    temp_hist_vars[key] = self.history_vars[key][:]

        fill_temp_hist_vars()

        # }}}

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
            for irhs, rhs in enumerate(rhss):
                cb.fence()
                n = rhs.order

                # {{{ compute AB coefficients

                # use:
                # Vandermonde^T * ab_coeffs = integrate(t_start, t_end, monomials)

                vdmt = var(name_gen("vdm_transpose"))
                cb(vdmt, array(n*n))

                coeff_rhs = var(name_gen("coeff_rhs"))
                cb(coeff_rhs, array(n))

                i = var(name_gen("vdm_i"))
                j = var(name_gen("vdm_j"))

                relevant_time_hist = temp_time_vars[comp_name, irhs][-n:][::-1]
                time_hist_var = var(name_gen("time_hist"))
                cb(time_hist_var, array(n))

                for ii in range(n):
                    cb(time_hist_var[ii], relevant_time_hist[ii] - self.t)

                cb(vdmt[i + j*n], time_hist_var[j]**i,
                    loops=[(i.name, 0, n), (j.name, 0, n)])

                t_start = self.dt * latest_state_substep / self.nsubsteps
                t_end = self.dt * isubstep / self.nsubsteps

                cb(coeff_rhs[i],
                        1/(i+1) * (t_end**(i+1) - t_start**(i+1)),
                        loops=[(i.name, 0, n)])

                ab_coeffs = var(name_gen("ab_coeffs"))
                cb(ab_coeffs, linear_solve(vdmt, coeff_rhs, n, 1))

                # }}}

                state_contrib_var = var(
                        name_gen(
                            "state_contrib_{comp_name}_rhs{irhs}"
                            .format(comp_name=comp_name, irhs=irhs)))

                cb(state_contrib_var,
                        _linear_comb(
                            [ab_coeffs[ii] for ii in range(n)],
                            temp_hist_vars[comp_name, irhs][-n:][::-1]))

                contribs.append(state_contrib_var)

                cb.fence()

            state_var = var(
                    name_gen(
                        "state_{comp_name}_sub{isubstep}"
                        .format(comp_name=comp_name, isubstep=isubstep)))

            state_expr = latest_state + sum(contribs)
            if self.state_filters[comp_index] is not None:
                state_expr = self.state_filters[comp_index](state_expr)
            cb(state_var, state_expr)

            states.append((isubstep, state_var))

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

            rhs_var = var(
                    name_gen(
                        "rhs_{comp_name}_rhs{irhs}_sub{isubstep}"
                        .format(comp_name=comp_name, irhs=irhs, isubstep=isubstep)))

            cb(rhs_var, var(rhs.func_name)(t=t_expr, **kwargs))

            temp_time_vars[comp_name, irhs].append(t_var)
            temp_hist_vars[comp_name, irhs].append(rhs_var)

            # }}}

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

                            if rhs.rhs_mode == rhs_mode.early_and_late:
                                temp_time_vars[comp_name, irhs].pop()
                                temp_hist_vars[comp_name, irhs].pop()

                            if rhs.rhs_mode in [
                                    rhs_mode.early_and_late, rhs_mode.late]:
                                update_hist(comp_idx, irhs, isubstep)

                            # }}}

                        if isubstep < self.nsubsteps:
                            # {{{ start up a new substep

                            if rhs.rhs_mode in [
                                    rhs_mode.early, rhs_mode.early_and_late]:
                                update_hist(comp_idx, irhs, isubstep + rhs.interval)

                            # }}}

        run_substep_loop()

        # }}}

        cb.fence()

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

    def generate(self):
        from dagrt.language import (TimeIntegratorCode, TimeIntegratorState,
                                      CodeBuilder)

        # Initialization state
        with CodeBuilder(label="initialization") as cb_init:
            self.emit_initialization(cb_init)

        # Primary state
        with CodeBuilder(label="primary") as cb_primary:
            self.emit_ab_method(cb_primary)

        with CodeBuilder(label="bootstrap") as cb_bootstrap:
            self.emit_rk_bootstrap(cb_bootstrap)

        states = {}
        states["initialization"] = TimeIntegratorState.from_cb(cb_init, "bootstrap")
        states["bootstrap"] = TimeIntegratorState.from_cb(cb_bootstrap, "bootstrap")
        states["primary"] = TimeIntegratorState.from_cb(cb_primary, "primary")

        return TimeIntegratorCode(
            instructions=cb_init.instructions | cb_bootstrap.instructions |
            cb_primary.instructions,
            states=states,
            initial_state="initialization")

        # }}}

# }}}


# {{{ two-rate compatibility shim

class TwoRateAdamsBashforthMethod(MultiRateAdamsBashforthMethod):
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
                "MultiRateAdamsBashforthMethod interface instead.",
                DeprecationWarning, stacklevel=2)

        if "S" in method:
            s2s_mode = rhs_mode.late
        else:
            s2s_mode = rhs_mode.early

        if "r" in method:
            s2s_mode = rhs_mode.early_and_late

        if "q" in method:
            s2f_interval = 1
        else:
            s2f_interval = step_ratio

        if "s" in method:
            f2s_mode = rhs_mode.early
        else:
            f2s_mode = rhs_mode.late

        if "f" in method:
            s2f_mode = rhs_mode.early
        else:
            s2f_mode = rhs_mode.late

        super(TwoRateAdamsBashforthMethod, self).__init__(
                order,
                component_names=("fast", "slow",),
                rhss=(
                    (
                        RHS(1, "<func>f2f", ("fast", "slow",)),
                        RHS(s2f_interval, "<func>s2f", ("fast", "slow",),
                            rhs_mode=s2f_mode),
                        ),
                    (
                        RHS(step_ratio, "<func>f2s", ("fast", "slow",),
                            rhs_mode=f2s_mode),
                        RHS(step_ratio, "<func>s2s", ("fast", "slow",),
                            rhs_mode=s2s_mode),
                        ),),

                state_filter_names=(fast_state_filter_name, slow_state_filter_name),

                # This is a hack to avoid having to change the 2RAB test
                # cases, which use these arguments
                component_arg_names=("f", "s"))

# }}}

# vim: foldmethod=marker
