#the ! /usr/bin/env python

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner, Matt Wala"

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

# avoid spurious: pytest.mark.parametrize is not callable
# pylint: disable=not-callable

import sys
import pytest

from leap.rk import ODE23MethodBuilder, ODE45MethodBuilder
import numpy as np
import numpy.linalg as la

import logging


logger = logging.getLogger(__name__)


# Run example with
# python test_step_matrix.py "test_step_matrix(ODE23MethodBuilder())"

@pytest.mark.parametrize("method", [
    ODE23MethodBuilder("y", use_high_order=False),
    ODE23MethodBuilder("y", use_high_order=True),
    ODE45MethodBuilder("y", use_high_order=False),
    ODE45MethodBuilder("y", use_high_order=True),
    ])
def test_step_matrix(method, show_matrix=True, show_dag=False):
    component_id = "y"
    code = method.generate()
    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)
    from dagrt.exec_numpy import NumpyInterpreter
    from leap.step_matrix import StepMatrixFinder

    from pymbolic import var

    # {{{ build matrix

    def rhs_sym(t, y):
        return var("lambda")*y

    finder = StepMatrixFinder(code, function_map={"<func>" + component_id: rhs_sym})

    mat = finder.get_phase_step_matrix("primary")

    if show_matrix:
        print("Variables: %s" % finder.variables)
        from pytools import indices_in_shape
        for i in indices_in_shape(mat.shape):
            print(i, mat[i])

    # }}}

    dt = 0.1
    lambda_ = -0.4

    def rhs(t, y):
        return lambda_*y

    interp = NumpyInterpreter(code, function_map={"<func>" + component_id: rhs})
    interp.set_up(t_start=0, dt_start=dt, context={component_id: 15})

    assert interp.next_phase == "initial"
    for _event in interp.run_single_step():
        pass
    assert interp.next_phase == "primary"

    start_values = np.array(
            [interp.context[v] for v in finder.variables])

    for _event in interp.run_single_step():
        pass
    assert interp.next_phase == "primary"

    stop_values = np.array(
            [interp.context[v] for v in finder.variables])

    from dagrt.expression import EvaluationMapper
    concrete_mat = EvaluationMapper({
        "lambda": lambda_,
        "<dt>": dt,
        }, {})(mat)

    stop_values_from_mat = concrete_mat.dot(start_values)

    rel_err = (
            la.norm(stop_values - stop_values_from_mat)
            /  # noqa: W504
            la.norm(stop_values))

    assert rel_err < 1e-12


def euler(component_id, show_dag):
    from leap.multistep import AdamsBashforthMethodBuilder

    method = AdamsBashforthMethodBuilder(component_id, 1, static_dt=True)
    code = method.generate()
    if show_dag:
        from dagrt.language import show_dependency_graph
        show_dependency_graph(code)
    return code


def test_step_matrix_vector_state(show_matrix=True, show_dag=False):
    from leap.step_matrix import StepMatrixFinder
    from pymbolic import var

    component_id = "y"
    code = euler(component_id, show_dag)
    J = np.diag([-3, -2, -1])  # noqa

    def rhs_sym(t, y):
        return J.dot(y)

    finder = StepMatrixFinder(
        code, function_map={"<func>" + component_id: rhs_sym},
        variables=["<state>" + component_id])

    mat = finder.get_phase_step_matrix("primary",
        shapes={"<state>" + component_id: 3})

    if show_matrix:
        print("Variables: %s" % finder.variables)
        from pytools import indices_in_shape
        for i in indices_in_shape(mat.shape):
            print(i, mat[i])

    # XXX: brittle
    dt = var("<dt>")
    true_mat = np.eye(3, dtype=np.object) + dt * J
    assert (mat == true_mat).all()


def test_step_matrix_fast_eval():
    from leap.step_matrix import StepMatrixFinder, fast_evaluator

    component_id = "y"
    code = euler(component_id, show_dag=False)
    J = np.diag([-3, -2, -1])  # noqa

    def rhs_sym(t, y):
        return J.dot(y)

    finder = StepMatrixFinder(
        code, function_map={"<func>" + component_id: rhs_sym},
        variables=["<state>" + component_id])

    mat = finder.get_phase_step_matrix("primary",
        shapes={"<state>" + component_id: 3})

    eval_mat = fast_evaluator(mat)
    assert (eval_mat({"<dt>": 1}) == np.diag([-2, -1, 0])).all()


def test_step_matrix_sparse():
    from leap.step_matrix import StepMatrixFinder, fast_evaluator
    from pymbolic import var

    component_id = "y"
    code = euler(component_id, show_dag=False)
    J = np.diag([-3, -2, -1])  # noqa

    def rhs_sym(t, y):
        return J.dot(y)

    finder = StepMatrixFinder(
        code, function_map={"<func>" + component_id: rhs_sym},
        variables=["<state>" + component_id])

    dt = var("<dt>")

    mat = finder.get_phase_step_matrix("primary",
        shapes={"<state>" + component_id: 3},
        sparse=True)

    assert mat.shape == (3, 3)
    # https://github.com/PyCQA/pylint/issues/3388
    assert mat.indices == [(0, 0), (1, 1), (2, 2)]  # pylint: disable=no-member
    true_mat = np.eye(3, dtype=np.object) + dt * J
    assert (mat.data == np.diag(true_mat)).all()

    eval_mat = fast_evaluator(mat, sparse=True)
    eval_result = eval_mat({"<dt>": 1})
    assert eval_result.shape == (3, 3)
    assert eval_result.indices == [(0, 0), (1, 1), (2, 2)]
    assert eval_result.data == [-2, -1, 0]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
