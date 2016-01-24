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


def _elide_yield_state(instructions):
    from dagrt.language import YieldState, Nop
    return [insn
            if not isinstance(insn, YieldState)
            else Nop(id=insn.id, depends_on=insn.depends_on)
            for insn in instructions]


def _elide_t_update(instructions):
    from dagrt.language import AssignExpression, Nop
    from pymbolic import var
    return [insn
            if (not isinstance(insn, AssignExpression)
                or insn.lhs != var("<t>"))
            else Nop(id=insn.id, depends_on=insn.depends_on)
            for insn in instructions]


def strang_splitting(dag1, dag2, stepping_state):
    from pymbolic.mapper.substitutor import make_subst_func, SubstitutionMapper

    all_states = frozenset(dag1.states) | frozenset(dag2.states)
    from dagrt.language import DAGCode, ExecutionState
    new_states = {}
    for state_name in all_states:
        state1 = dag1.states.get(state_name)
        state2 = dag2.states.get(state_name)

        if state_name == stepping_state:
            assert state1 is not None
            assert state2 is not None

            from pymbolic import var
            dt_half = SubstitutionMapper(
                    make_subst_func({"<dt>": var("<dt>") / 2}))

            state1_half_dt = [
                        insn.map_expressions(dt_half)
                        for insn in state1.instructions]

            if state1.next_state != state2.next_state:
                raise ValueError("DAGs don't agree on default "
                        "state transition out of state '%s'"
                        % state_name)

            s2_name = state_name + "_s2"
            s3_name = state_name + "_s3"

            assert s2_name not in all_states
            assert s3_name not in all_states

            new_states[state_name] = ExecutionState(
                    next_state=s2_name,
                    depends_on=state1.depends_on,
                    instructions=_elide_yield_state(state1_half_dt))
            new_states[s2_name] = ExecutionState(
                    next_state=s3_name,
                    depends_on=state2.depends_on,
                    instructions=_elide_t_update(
                        _elide_yield_state(state2.instructions)))
            new_states[s3_name] = ExecutionState(
                    next_state=state1.next_state,
                    depends_on=state1.depends_on,
                    instructions=state1_half_dt)
        else:
            from dagrt.transform import fuse_two_states
            new_states[state_name] = fuse_two_states(state_name, state1, state2)

    if dag1.initial_state != dag2.initial_state:
        raise ValueError("DAGs don't agree on initial state")

    return DAGCode(new_states, dag1.initial_state)
