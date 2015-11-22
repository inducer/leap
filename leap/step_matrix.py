from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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


from dagrt.expression import EvaluationMapper
import numpy as np
from dagrt.exec_numpy import FailStepException


# {{{ step matrix finder

class StepMatrixFinder(object):
    """Constructs a step matrix on-the-fly while interpreting code.

    Assumes that all function evaluations occur as the root node of
    a separate assignment instruction.
    """

    def __init__(self, code, function_map, variables=None):
        self.code = code

        self.function_map = function_map

        if variables is None:
            variables = self._get_state_variables()
        self.variables = variables

        from dagrt.language import ExecutionController
        self.exec_controller = ExecutionController(code)
        self.context = {}

        self.eval_mapper = EvaluationMapper(self.context, self.function_map)

    def _get_state_variables(self):
        """Extract all state-related variables from the code."""
        all_var_ids = set()
        for inst in self.code.instructions:
            all_var_ids |= inst.get_assignees()
            all_var_ids |= inst.get_read_variables()
        all_state_vars = []
        for var_name in all_var_ids:
            if var_name.startswith('<p>') or var_name.startswith('<state>'):
                all_state_vars.append(var_name)
        all_state_vars.sort()
        return all_state_vars

    def get_state_step_matrix(self, state_name):
        state = self.code.states[state_name]

        from pymbolic import var

        initial_vars = []

        self.context.clear()
        for vname in self.variables:
            iv = self.context[vname] = var(vname+"_0")
            initial_vars.append(iv)

        self.context["<dt>"] = var("<dt>")
        self.context["<t>"] = 0

        self.exec_controller.reset()
        self.exec_controller.update_plan(state.depends_on)
        for event in self.exec_controller(self):
            pass

        from pymbolic.mapper.differentiator import DifferentiationMapper

        nv = len(self.variables)
        step_matrix = np.zeros((nv, nv), dtype=np.object)
        for i, v in enumerate(self.variables):
            for j, iv in enumerate(initial_vars):
                step_matrix[i][j] = DifferentiationMapper(iv)(self.context[v])
        return step_matrix

    def evaluate_condition(self, insn):
        if insn.condition is not True:
            raise RuntimeError("matrices don't represent conditionals well, "
                "so StepMatrixFinder cannot support them")
        return True

    # {{{ exec methods

    def exec_AssignExpression(self, insn):
        self.context[insn.assignee] = self.eval_mapper(insn.expression)

    def exec_AssignFunctionCall(self, insn):
        results = self.eval_mapper(insn.as_expression())

        if len(insn.assignees) == 1:
            results = (results,)

        assert len(results) == len(insn.assignees)

        for assignee, res in zip(insn.assignees, results):
            self.context[assignee] = res

    def exec_Nop(self, insn):
        pass

    def exec_YieldState(self, insn):
        pass

    def exec_Raise(self, insn):
        raise insn.error_condition(insn.error_message)

    def exec_FailStep(self, insn):
        raise FailStepException()

    def exec_StateTransition(self, insn):
        pass

    # }}}

# }}}

# vim: foldmethod=marker
