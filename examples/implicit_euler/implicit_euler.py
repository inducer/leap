#!/usr/bin/env python
"""Implicit Euler timestepper"""

from leap import MethodBuilder
from dagrt.language import DAGCode, CodeBuilder
from pymbolic import var
from pymbolic.primitives import CallWithKwargs

__copyright__ = "Copyright (C) 2014 Matt Wala"

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


class ImplicitEulerMethodBuilder(MethodBuilder):
    """
    Context:
       state: The value that is integrated
       rhs_func: The right hand side function
    """

    SOLVER_EXPRESSION_ID = 0

    def __init__(self, component_id):
        self.component_id = component_id
        self.dt = var("<dt>")
        self.t = var("<t>")
        self.state = var("<state>" + component_id)
        self.rhs_func = var("<func>" + component_id)

    def generate(self, solver_hook):
        """Return code that implements the implicit Euler method for the single
        state component supported."""

        with CodeBuilder(name="primary") as cb:
            self._make_primary(cb)

        code = DAGCode.from_phases_list(
                [cb.as_execution_phase(next_phase="primary")],
                initial_phase="primary")

        from leap.implicit import replace_AssignImplicit

        return replace_AssignImplicit(code, {self.SOLVER_EXPRESSION_ID: solver_hook})

    def _make_primary(self, builder):
        """Add code to drive the primary stage."""

        solve_component = var("next_state")
        solve_expression = solve_component - self.state - \
                           self.dt * CallWithKwargs(
                               function=self.rhs_func,
                               parameters=(),
                               kw_parameters={
                                   "t": self.t + self.dt,
                                   self.component_id: solve_component
                               })

        builder.assign_implicit_1(self.state, solve_component,
                                solve_expression, self.state,
                                self.SOLVER_EXPRESSION_ID)

        builder.yield_state(self.state, self.component_id,
                            self.t + self.dt, "final")
        builder.assign(self.t, self.t + self.dt)
