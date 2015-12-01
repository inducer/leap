class FunctionMarshaller(object):
    """
    A wrapper that allows pickling and unpickling of functions.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        from marshal import dumps
        return (dumps(self.func.func_code), self.func.func_name)

    def __setstate__(self, state):
        import marshal
        import types
        code = marshal.loads(state[0])
        self.func = types.FunctionType(code, globals(), state[1])


def fast_evaluator(matrix, marshalled=False):
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
            self.name_map = KeyToUniqueNameMap(forced_prefix="local")

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
    code.append("wrapper = FunctionMarshaller(evaluate)")
    exec_locals = {"FunctionMarshaller": FunctionMarshaller}
    exec_globals = {}
    exec("\n".join(code), exec_globals, exec_locals)
    return exec_locals["wrapper"]
