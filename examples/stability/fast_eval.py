def fast_evaluator(matrix):
    """
    Generates a function to evaluate a step matrix quickly.
    The input should be numpy array with pymbolic expression entries.
    """
    from dagrt.codegen.expressions import PythonExpressionMapper
    from dagrt.codegen.utils import KeyToUniqueNameMap
    from dagrt.function_registry import base_function_registry
    from dagrt.utils import get_variables
    from pymbolic import var

    class NameManager(object):

        def __init__(self):
            self.name_map = KeyToUniqueNameMap()

        def __getitem__(self, key):
            return self.name_map.get_or_make_name_for_key(key)

    expr_mapper = PythonExpressionMapper(NameManager(), base_function_registry)
    code = []
    code.append("def evaluate(vars):")
    code.append(" import numpy")

    all_vars = get_variables(matrix)
    for var_name in all_vars:
        code.append(" {var} = vars[\"{var_name}\"]".format(
            var=expr_mapper(var(var_name)), var_name=var_name))

    def descend_matrix(index):
        depth = len(index)
        if depth == len(matrix.shape):
            return expr_mapper(matrix.item(*index))
        return "[" + ",".join(descend_matrix(index + [i])
                              for i in range(matrix.shape[depth])) + "]"

    code.append(" return numpy.array({matrix}, dtype=numpy.complex128)"
                .format(matrix=descend_matrix([])))
    exec_locals = {}
    exec_globals = {}
    exec("\n".join(code), exec_globals, exec_locals)
    return exec_locals["evaluate"]
