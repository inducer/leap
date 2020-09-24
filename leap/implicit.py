"""Implicit solver utilities"""

__copyright__ = """
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


def replace_AssignImplicit(dag, solver_hooks):
    """
    :arg dag: The :class:`DAGCode` instance
    :arg solver_hooks: Either a callable, or a map from solver names to
        functions that generate solver calls.

        A solver hook should have the signature::

            def solver_hook(expr, var, id, **kwargs):

        where:
         * *expr* is the expression passed to the AssignImplicit instruction
         * *var* is the name of the unknown
         * *id* is the *solver_id* field of the AssignImplicit instruction
         * any other arguments are passed in *kwargs*.
    """

    if callable(solver_hooks):
        hook = solver_hooks
        from collections import defaultdict
        solver_hooks = defaultdict(lambda: hook)

    from dagrt.language import Assign, AssignImplicit

    new_phases = {}

    for phase_name, phase in dag.phases.items():
        new_statements = []

        for stmt in phase.statements:

            if not isinstance(stmt, AssignImplicit):
                new_statements.append(stmt)
                continue

            if len(stmt.assignees) != 1:
                from dagrt.utils import TODO
                raise TODO("Implement lowering for AssignImplicit statements "
                           "returning multiple values.")

            expression = stmt.expressions[0]
            solve_variable = stmt.solve_variables[0]
            solver_id = stmt.solver_id
            other_params = stmt.other_params

            solver_hook = solver_hooks[stmt.solver_id]
            solver_expression = solver_hook(expression, solve_variable,
                                            solver_id, **other_params)

            new_statements.append(
                Assign(
                    assignee=stmt.assignees[0],
                    assignee_subscript=(),
                    expression=solver_expression,
                    id=stmt.id,
                    condition=stmt.condition,
                    depends_on=stmt.depends_on))

        new_phases[phase_name] = phase.copy(statements=new_statements)

    return dag.copy(phases=new_phases)
