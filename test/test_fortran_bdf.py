"""
Fortran version of the reactors. All we need for now is to
try running a Fortran code generator with the method.
"""


def test_vs_scipy():

    from leap.multistep import AdaptiveBDFMethodBuilder

    rtol = 1e-4
    atol = 1e-16

    method = AdaptiveBDFMethodBuilder("y", use_high_order=True, ndf=False,
                                      atol=atol, rtol=rtol, max_dt_growth=10,
                                      min_dt_shrinkage=0.2)

    code = method.generate()

    import dagrt.codegen.fortran as f

    # For now, we will register simple dummy rhs and solver functions.
    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs,
            register_function, UserType)
    rhs_function = "<func>y"
    solver_function = "<func>solver"
    freg = register_ode_rhs(base_function_registry, "y",
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))
    freg = register_function(freg, solver_function,
            ("t", "sub_y", "coeff", "guess",),
            result_names=("result",), result_kinds=(UserType("y"),))
    freg = freg.register_codegen(solver_function, "fortran",
            f.CallCode("""
                ${result} = -2*${sub_y}
                """))

    def solver_hook(solve_expr, solve_var, solver_id, guess):
        from dagrt.expression import match, substitute

        pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                        pre_match={"unk": solve_var})
        pieces["guess"] = guess
        return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})

    # Make some dummy thing to use to troubleshoot...user array
    # of size one to start with...
    codegen = f.CodeGenerator("BDFMethod",
                              user_type_map={"y": f.ArrayType((2,),
                                  f.BuiltinType("real (kind=8)"),)},
                              function_registry=freg,
                              module_preamble="""
                              ! lines copied to the start of the module, e.g. to say:
                              ! use ModStuff
                              """)

    code_str = codegen(code)
    text_file = open("test.f90", "w")
    text_file.write(code_str)
    text_file.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
