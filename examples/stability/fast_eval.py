from __future__ import division, print_function


def fast_evaluator(matrix):
    """
    Generate a function to evaluate a step matrix quickly.
    The input should be numpy array whose entries are pymbolic expressions.
    """
    # First, rename variables in the matrix to names that are acceptable Python
    # identifiers. We make use of dagrt's KeyToUniqueNameMap.
    from dagrt.codegen.utils import KeyToUniqueNameMap
    name_map = KeyToUniqueNameMap(forced_prefix="matrix")

    def make_identifier(symbol):
        from pymbolic import var
        assert isinstance(symbol, var)
        return var(name_map.get_or_make_name_for_key(symbol.name))

    from pymbolic.mapper.substitutor import SubstitutionMapper
    matrix = SubstitutionMapper(make_identifier)(matrix)

    # Compile the matrix.
    orig_varnames = sorted(key for key in name_map)
    renamed_varnames = [name_map.get_or_make_name_for_key(key)
                        for key in orig_varnames]
    from pymbolic import compile
    compiled_matrix = compile(matrix, renamed_varnames)

    # functools.partial ensures the resulting object is picklable.
    from functools import partial
    return partial(_eval_compiled_matrix, compiled_matrix, orig_varnames)


def _eval_compiled_matrix(compiled_matrix, var_order, var_assignments):
    """
    :arg compiled_matrix: A compiled pymbolic expression
    :arg var_order: A list of keys. Arguments are passed in this order
    :arg var_assignments: A dictionary, mapping keys in `var_order` to values
    :return: The evaluted matrix as a numpy array
    """
    arguments = [var_assignments[name] for name in var_order]
    return compiled_matrix(*arguments)
