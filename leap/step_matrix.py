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

from collections import namedtuple
from dagrt.expression import EvaluationMapper
import numpy as np
from dagrt.exec_numpy import FailStepException
from pymbolic.interop.maxima import MaximaStringifyMapper
from pytools import Record

__doc__ = """

.. autoclass:: StepMatrixFinder
.. autofunction:: fast_evaluator
"""


class SparseStepMatrix(Record):

    def __init__(self, shape, indices, data):
        Record.__init__(self, shape=shape, indices=indices, data=data)


class LeapMaximaStringifyMapper(MaximaStringifyMapper):
    def map_variable(self, expr, prec):
        s = expr.name
        s = s.replace("<", "_")
        s = s.replace(">", "_")
        s = s.strip("_")
        return s


# {{{ step matrix finder

class StepMatrixFinder:
    """Constructs a step matrix on-the-fly while interpreting code.

    Assumes that all function evaluations occur as the root node of
    a separate assignment statement.
    """

    def __init__(self, code, function_map, variables=None,
            exclude_variables=None):
        self.code = code

        if exclude_variables is None:
            exclude_variables = []

        self.exclude_variables = exclude_variables

        from dagrt.builtins_python import builtins

        # Ensure none of the names in the function map conflict with the
        # builtins.
        assert not set(builtins) & set(function_map)

        self.function_map = builtins.copy()
        self.function_map.update(function_map)

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
        for phase in self.code.phases.values():
            for inst in phase.statements:
                all_var_ids |= inst.get_written_variables()
                all_var_ids |= inst.get_read_variables()

        all_state_vars = []
        for var_name in all_var_ids:
            if (
                    (var_name.startswith("<p>")
                        or var_name.startswith("<state>"))
                    and var_name not in self.exclude_variables):
                all_state_vars.append(var_name)

        all_state_vars.sort()
        return all_state_vars

    VectorComponent = namedtuple("VectorComponent", "name, index")

    def run_symbolic_step(self, phase_name, shapes=None):
        """
        `shapes` maps variable names to vector lengths.
        """
        if shapes is None:
            shapes = {}
        phase = self.code.phases[phase_name]

        from pymbolic import var

        self.context.clear()
        from pymbolic.primitives import make_sym_vector

        # Includes variables expanded into vector components
        components = []
        # Initial values, as variables / subscripts. Matched with "components"
        initial_vals = []
        for vname in self.variables:
            if vname in shapes and shapes[vname] > 1:
                ival = make_sym_vector(vname+"_0", shapes[vname])
                initial_vals.extend(ival)
                names = [self.VectorComponent(vname, i) for i in range(len(ival))]
                components.extend(names)
            else:
                ival = var(vname+"_0")
                initial_vals.append(ival)
                components.append(vname)
            self.context[vname] = ival

        self.context["<dt>"] = var("<dt>")
        self.context["<t>"] = 0

        self.exec_controller.reset()
        self.exec_controller.update_plan(phase, phase.depends_on)
        for _event in self.exec_controller(phase, self):
            pass

        return components, initial_vals

    def get_maxima_expressions(self, phase_name, shapes=None):
        if shapes is None:
            shapes = {}
        components, initial_vals = self.run_symbolic_step(phase_name, shapes)

        lines = []

        msm = LeapMaximaStringifyMapper()

        def msm_expr_list(name, exprs):
            lines.append(f"{name}: [")
            for i, expr in enumerate(exprs):
                line = "    "+msm(expr)

                if i + 1 != len(initial_vals):
                    line += ","
                else:
                    line += "];"
                lines.append(line)
            lines.append("")

        msm_expr_list("initial", initial_vals)

        exprs = []
        for v in components:
            # Get the expression for v.
            if isinstance(v, self.VectorComponent):
                expr = self.context[v.name][v.index]
            else:
                expr = self.context[v]

            exprs.append(expr)

        msm_expr_list("after_step", exprs)

        return "\n".join(lines)

    def get_phase_step_matrix(self, phase_name, shapes=None, sparse=False):
        """
        `shapes` maps variable names to vector lengths.

        `sparse` controls whether the output is sparse or dense. When
             `sparse=True`, returns a SparseStepMatrix.
             Otherwise returns a numpy object array.
        """

        if shapes is None:
            shapes = {}

        components, initial_vals = self.run_symbolic_step(phase_name, shapes)

        from pymbolic.mapper.differentiator import DifferentiationMapper
        from pymbolic.mapper.dependency import DependencyMapper
        dependencies = DependencyMapper()

        nv = len(components)
        shape = (nv, nv)
        if not sparse:
            step_matrix = np.zeros(shape, dtype=np.object)
        else:
            indices = []
            data = []

        iv_to_index = {iv: i for i, iv in enumerate(initial_vals)}
        for i, v in enumerate(components):
            # Get the expression for v.
            if isinstance(v, self.VectorComponent):
                expr = self.context[v.name][v.index]
            else:
                expr = self.context[v]

            # Selectively evaluate the derivative only for components that are
            # actually present in the expression. This takes advantage of
            # sparsity.
            component_vars = dependencies(expr)
            for iv in component_vars:
                if iv not in iv_to_index:
                    continue
                j = iv_to_index[iv]
                entry = DifferentiationMapper(iv)(expr)

                if not sparse:
                    step_matrix[i, j] = entry
                else:
                    indices.append((i, j))
                    data.append(entry)

        if not sparse:
            return step_matrix
        else:
            return SparseStepMatrix(shape, indices, data)

    def evaluate_condition(self, stmt):
        if stmt.condition is not True:
            raise RuntimeError("matrices don't represent conditionals well, "
                "so StepMatrixFinder cannot support them")
        return True

    # {{{ exec methods

    def exec_Assign(self, stmt):
        self.context[stmt.assignee] = self.eval_mapper(stmt.expression)

    def exec_AssignFunctionCall(self, stmt):
        results = self.eval_mapper(stmt.as_expression())

        if len(stmt.assignees) == 1:
            results = (results,)

        assert len(results) == len(stmt.assignees)

        for assignee, res in zip(stmt.assignees, results):
            self.context[assignee] = res

    def exec_Nop(self, stmt):
        pass

    def exec_YieldState(self, stmt):
        pass

    def exec_Raise(self, stmt):
        raise stmt.error_condition(stmt.error_message)

    def exec_FailStep(self, stmt):
        raise FailStepException()

    def exec_SwitchPhase(self, stmt):
        pass

    # }}}

# }}}

# {{{ fast evaluation for step matrices


def fast_evaluator(matrix, sparse=False):
    """
    Generate a function to evaluate a step matrix quickly.
    The input comes from StepMatrixFinder.
    """
    # First, rename variables in the matrix to names that are acceptable Python
    # identifiers. We make use of dagrt's KeyToUniqueNameMap.
    from dagrt.codegen.utils import KeyToUniqueNameMap
    name_map = KeyToUniqueNameMap(forced_prefix="matrix")

    def make_identifier(symbol):
        from pymbolic import var
        assert isinstance(symbol, var)
        return var(name_map.get_or_make_name_for_key(symbol.name))

    def get_var_order_from_name_map():
        order = sorted(name_map)
        return (order,
            [name_map.get_or_make_name_for_key(key) for key in order])

    from pymbolic.mapper.substitutor import SubstitutionMapper

    substitutor = SubstitutionMapper(make_identifier)

    from pymbolic import compile
    # functools.partial ensures the resulting object is picklable.
    from functools import partial

    if sparse:
        data = [substitutor(entry) for entry in matrix.data]
        var_order, renamed_vars = get_var_order_from_name_map()
        compiled_entries = [compile(entry, renamed_vars) for entry in data]
        compiled_matrix = matrix.copy(data=compiled_entries)
    else:
        matrix = substitutor(matrix)
        var_order, renamed_vars = get_var_order_from_name_map()
        compiled_matrix = compile(matrix, renamed_vars)

    return partial(_eval_compiled_matrix, compiled_matrix, var_order)


def _eval_compiled_matrix(compiled_matrix, var_order, var_assignments):
    """
    :arg compiled_matrix: Either a SparseStepMatrix or a compiled pymbolic expression
    :arg var_order: A list of keys. Arguments are passed in this order
    :arg var_assignments: A dictionary, mapping keys in `var_order` to values
    :return: The evaluted matrix as a numpy array
    """
    arguments = [var_assignments[name] for name in var_order]
    if isinstance(compiled_matrix, SparseStepMatrix):
        evaluated_data = [entry(*arguments) for entry in compiled_matrix.data]
        return compiled_matrix.copy(data=evaluated_data)
    else:
        return compiled_matrix(*arguments)

# }}}

# vim: foldmethod=marker
