#! /usr/bin/env python
from __future__ import division, with_statement, print_function

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


__doc__ = """
.. autofunction:: strang_splitting

"""


def _elide_yield_state(instructions):
    from dagrt.language import YieldState, Nop
    return [insn
            if not isinstance(insn, YieldState)
            else Nop(id=insn.id, depends_on=insn.depends_on)
            for insn in instructions]


def _update_t_by_dt_factor(factor, instructions):
    from dagrt.language import AssignExpression, Nop
    from pymbolic import var
    from pymbolic.mapper.substitutor import make_subst_func, SubstitutionMapper

    mapper = SubstitutionMapper(
        make_subst_func({"<dt>": factor * var("<dt>")}))

    def updater(insn):
        if factor == 0:
            return Nop(id=insn.id, depends_on=insn.depends_on)
        return insn.map_expressions(mapper)

    return [insn
            if (not isinstance(insn, AssignExpression)
                or insn.lhs != var("<t>"))
            else updater(insn)
            for insn in instructions]


def strang_splitting(dag1, dag2, stepping_phase):
    """Given two time advancement routines (in *dag1* and *dag2*), returns a
    single second-order accurate time advancement routine representing the sum
    of both of those advancements.

    :arg dag1: a :class:`dagrt.language.DAGCode`
    :arg dag2: a :class:`dagrt.language.DAGCode`
    :arg stepping_phase: the name of the phase in *dag1* and *dag2* that carries
        out time stepping to which Strang splitting is to be applied.
    :returns: a :class:`dagrt.language.DAGCode`
    """

    from pymbolic.mapper.substitutor import make_subst_func, SubstitutionMapper

    # {{{ disambiguate

    id1 = dag1.existing_var_names()
    id2 = dag1.existing_var_names()

    from pytools import UniqueNameGenerator
    vng = UniqueNameGenerator(id1 | id2)

    from pymbolic import var
    subst2 = {}
    for clash in id1 & id2:
        if not clash.startswith("<") or clash.startswith("<p>"):
            unclash = vng(clash)
            subst2[clash] = var(unclash)

    subst2_mapper = SubstitutionMapper(make_subst_func(subst2))

    # }}}

    all_phases = frozenset(dag1.phases) | frozenset(dag2.phases)
    from dagrt.language import DAGCode, ExecutionPhase
    new_phases = {}
    for phase_name in all_phases:
        phase1 = dag1.phases.get(phase_name)
        phase2 = dag2.phases.get(phase_name)

        substed_s2_insns = [
                insn.map_expressions(subst2_mapper)
                for insn in phase2.instructions]

        if phase_name == stepping_phase:
            assert phase1 is not None
            assert phase2 is not None

            from pymbolic import var
            dt_half = SubstitutionMapper(
                    make_subst_func({"<dt>": var("<dt>") / 2}))

            phase1_half_dt = [
                        insn.map_expressions(dt_half)
                        for insn in phase1.instructions]

            if phase1.next_phase != phase2.next_phase:
                raise ValueError("DAGs don't agree on default "
                        "phase transition out of phase '%s'"
                        % phase_name)

            s2_name = phase_name + "_s2"
            s3_name = phase_name + "_s3"

            assert s2_name not in all_phases
            assert s3_name not in all_phases

            """
            du/dt = A + B
            Time interval is [0,1]
            1. Starting with u0, solve du / dt = A from t = 0 to 1/2, get u1
            2. Starting with u1, solve du / dt = B from t = 0 to 1, get u2
            3. Starting with u2, solve du / dt = A from t = 1/2 to 1, get u3
            4. Return u3
            """
            new_phases[phase_name] = ExecutionPhase(
                    next_phase=s2_name,
                    depends_on=phase1.depends_on,
                    instructions=(
                        _update_t_by_dt_factor(0,
                            _elide_yield_state(
                                phase1_half_dt))))
            new_phases[s2_name] = ExecutionPhase(
                    next_phase=s3_name,
                    depends_on=phase2.depends_on,
                    instructions=(
                        _update_t_by_dt_factor(1/2,
                            _elide_yield_state(
                                substed_s2_insns))))
            new_phases[s3_name] = ExecutionPhase(
                    next_phase=phase1.next_phase,
                    depends_on=phase1.depends_on,
                    instructions=phase1_half_dt)
        else:
            from dagrt.transform import fuse_two_phases
            new_phases[phase_name] = fuse_two_phases(phase_name,
                    phase1,
                    phase2.copy(instructions=substed_s2_insns))

    if dag1.initial_phase != dag2.initial_phase:
        raise ValueError("DAGs don't agree on initial phase")

    return DAGCode(new_phases, dag1.initial_phase)
