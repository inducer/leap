#! /usr/bin/env python
from __future__ import division, with_statement, print_function

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

import sys
import pytest

import dagrt.codegen.fortran as f
from leap.rk import ODE23Method, ODE45Method, LSRK4Method

from leap.ab.multirate import TwoRateAdamsBashforthMethod

from dagrt.utils import run_fortran


#skip = pytest.mark.skipif(True, reason="not fully implemented")


def read_file(rel_path):
    from os.path import join, abspath, dirname
    path = join(abspath(dirname(__file__)), rel_path)
    with open(path, "r") as inf:
        return inf.read()


# {{{ test rk methods

@pytest.mark.parametrize(("min_order", "stepper"), [
    (2, ODE23Method("y", use_high_order=False)),
    (3, ODE23Method("y", use_high_order=True)),
    (4, ODE45Method("y", use_high_order=False)),
    (5, ODE45Method("y", use_high_order=True)),
    (4, LSRK4Method("y")),
    ])
def test_rk_codegen(min_order, stepper):
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk.f90", read_file("test_rk.f90").replace(
            "MIN_ORDER", str(min_order - 0.3)+"d0")),
        ])

# }}}


# {{{ test fancy codegen

def test_rk_codegen_fancy():
    """Test whether Fortran code generation with lots of fancy features for the
    Runge-Kutta timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'
    state_filter_name = 'state_filter_y'

    stepper = ODE23Method(component_id, use_high_order=True,
            state_filter_name=state_filter_name)

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs,
            register_function, UserType)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                <%

                igrid = declare_new("integer", "igrid")
                i = declare_new("integer", "i")

                %>

                do ${igrid} = 1, region%n_grids
                  do ${i} = 1, region%n_grid_dofs(${igrid})
                    ${result}(${igrid})%conserved_var(${i}) = &
                     -2*${y}(${igrid})%conserved_var(${i})
                  end do
                end do

                """))
    freg = register_function(freg, "notify_pre_state_update", ("updated_component",))
    freg = freg.register_codegen("notify_pre_state_update", "fortran",
            f.CallCode("""
                write(*,*) 'before state update'
                """))
    freg = register_function(
            freg, "notify_post_state_update", ("updated_component",))
    freg = freg.register_codegen("notify_post_state_update", "fortran",
            f.CallCode("""
                write(*,*) 'after state update'
                """))

    freg = register_function(freg, "<func>"+state_filter_name, ("y",),
            result_names=("result",), result_kinds=(UserType("y"),))
    freg = freg.register_codegen("<func>"+state_filter_name, "fortran",
            f.CallCode("""
                ! mess with state
                <%

                igrid = declare_new("integer", "igrid")
                i = declare_new("integer", "i")

                %>

                do ${igrid} = 1, region%n_grids
                  do ${i} = 1, region%n_grid_dofs(${igrid})
                    ${result}(${igrid})%conserved_var(${i}) = &
                     0.95*${y}(${igrid})%conserved_var(${i})
                  end do
                end do

                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    "region%n_grids",
                    index_vars="igrid",
                    element_type=f.StructureType(
                        "sim_grid_state_type",
                        (
                            ("conserved_var", f.PointerType(
                                f.ArrayType(
                                    ("region%n_grid_dofs(igrid)",),
                                    f.BuiltinType('real (kind=8)')))),
                        )))
                },
            function_registry=freg,
            module_preamble="""
                use sim_types
                use timing

                """,
            call_before_state_update="notify_pre_state_update",
            call_after_state_update="notify_post_state_update",
            extra_arguments="region",
            extra_argument_decl="""
                type(region_type), pointer :: region
                """,
            parallel_do_preamble="!dir$ simd",
            emit_instrumentation=True,
            timing_function="get_time")

    code_str = codegen(code)
    print(code_str)

    run_fortran([
        ("sim_types.f90", read_file("sim_types.f90")),
        ("timing.f90", read_file("timing.f90")),
        ("rkmethod.f90", code_str),
        ("test_fancy_rk.f90", read_file("test_fancy_rk.f90")),
        ])

# }}}


@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
@pytest.mark.parametrize("method_name", TwoRateAdamsBashforthMethod.methods)
def test_multirate_codegen(min_order, method_name):
    from leap.ab.multirate import TwoRateAdamsBashforthMethod

    stepper = TwoRateAdamsBashforthMethod(
            method_name, min_order, 4,
            slow_state_filter_name="slow_filt",
            fast_state_filter_name="fast_filt")

    code = stepper.generate()

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs,
            UserType, register_function)

    freg = base_function_registry
    for func_name in [
            "<func>s2s",
            "<func>f2s",
            "<func>s2f",
            "<func>f2f",
            ]:
        component_id = {
                "s": "slow",
                "f": "fast",
                }[func_name[-1]]
        freg = register_ode_rhs(freg, identifier=func_name,
                component_id=component_id,
                input_component_ids=("slow", "fast"),
                input_component_names=("s", "f"))

    freg = freg.register_codegen("<func>s2f", "fortran",
        f.CallCode("""
            ${result} = (sin(2*${t}) - 1)*${s}
            """))
    freg = freg.register_codegen("<func>f2s", "fortran",
      f.CallCode("""
          ${result} = (sin(2*${t}) + 1)*${f}
          """))
    freg = freg.register_codegen("<func>f2f", "fortran",
      f.CallCode("""
          ${result} = cos(2*${t})*${f}
          """))
    freg = freg.register_codegen("<func>s2s", "fortran",
      f.CallCode("""
          ${result} = -cos(2*${t})*${s}
          """))

    freg = register_function(freg, "<func>slow_filt", ("arg",),
            result_names=("result",), result_kinds=(UserType("slow"),))
    freg = freg.register_codegen("<func>slow_filt", "fortran",
            f.CallCode("""
                ! mess with state
                ${result} = ${arg}
                """))

    freg = register_function(freg, "<func>fast_filt", ("arg",),
            result_names=("result",), result_kinds=(UserType("fast"),))
    freg = freg.register_codegen("<func>fast_filt", "fortran",
            f.CallCode("""
                ! mess with state
                ${result} = ${arg}
                """))

    codegen = f.CodeGenerator(
            'MRAB',
            ode_component_type_map={
                "slow": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                "fast": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    if 0:
        with open("abmethod.f90", "wt") as outf:
            outf.write(code_str)

    if min_order == 3:
        num_trips_one = 200
        num_trips_two = 300
    else:
        num_trips_one = 100
        num_trips_two = 150

    run_fortran([
        ("abmethod.f90", code_str),
        ("test_mrab.f90", (
            read_file("test_mrab.f90")
            .replace("MIN_ORDER", str(min_order - 0.3)+"d0")
            .replace("NUM_TRIPS_ONE", str(num_trips_one))
            .replace("NUM_TRIPS_TWO", str(num_trips_two)))),
        ],
        fortran_options=["-llapack", "-lblas"])


def test_adaptive_rk_codegen():
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = ODE45Method(component_id, use_high_order=False, rtol=1e-6)

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result}(1) = ${y}(2)
                ${result}(2) = -30*((${y}(1))**2 - 1)*${y}(2) - ${y}(1)
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                "y": f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                },
            function_registry=freg)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk_adaptive.f90", read_file("test_rk_adaptive.f90")),
        ])


def test_adaptive_rk_codegen_error():
    """Test whether Fortran code generation for the Runge-Kutta
    timestepper works.
    """

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = ODE45Method(component_id, use_high_order=False, atol=1e-6)

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'RKMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg)

    run_fortran([
        ("rkmethod.f90", codegen(code)),
        ("test_rk_adaptive_error.f90", read_file("test_rk_adaptive_error.f90")),
        ])


@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
def test_singlerate_squarewave(min_order):
    from leap.ab import AdamsBashforthMethod

    component_id = 'y'
    rhs_function = '<func>y'

    stepper = AdamsBashforthMethod("y", min_order)

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)
    freg = register_ode_rhs(base_function_registry, component_id,
                            identifier=rhs_function)
    freg = freg.register_codegen(rhs_function, "fortran",
            f.CallCode("""
                ${result} = -2*${y}
                """))

    code = stepper.generate()

    codegen = f.CodeGenerator(
            'ABMethod',
            ode_component_type_map={
                component_id: f.ArrayType(
                    (2,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    run_fortran([
        ("abmethod.f90", code_str),
        ("test_ab_squarewave.f90", read_file("test_ab_squarewave.f90").replace(
            "MIN_ORDER", str(min_order - 0.3)+"d0")),
        ],
        fortran_options=["-llapack", "-lblas"])


@pytest.mark.parametrize("method_name", TwoRateAdamsBashforthMethod.methods)
@pytest.mark.parametrize("min_order", [2, 3, 4, 5])
def test_multirate_squarewave(min_order, method_name):
    from pytools import DictionaryWithDefault

    orders = DictionaryWithDefault(lambda x: min_order)

    stepper = TwoRateAdamsBashforthMethod(method_name, orders, 4)

    code = stepper.generate()

    from dagrt.function_registry import (
            base_function_registry, register_ode_rhs)

    freg = base_function_registry
    for func_name in [
            "<func>s2s",
            "<func>f2s",
            "<func>s2f",
            "<func>f2f",
            ]:
        component_id = {
                "s": "slow",
                "f": "fast",
                }[func_name[-1]]
        freg = register_ode_rhs(freg, identifier=func_name,
                component_id=component_id,
                input_component_ids=("slow", "fast"),
                input_component_names=("s", "f"))

    freg = freg.register_codegen("<func>s2f", "fortran",
        f.CallCode("""
            ${result} = (sin(2*${t}) - 1)*${s}
            """))
    freg = freg.register_codegen("<func>f2s", "fortran",
      f.CallCode("""
          ${result} = (sin(2*${t}) + 1)*${f}
          """))
    freg = freg.register_codegen("<func>f2f", "fortran",
      f.CallCode("""
          ${result} = cos(2*${t})*${f}
          """))
    freg = freg.register_codegen("<func>s2s", "fortran",
      f.CallCode("""
          ${result} = -cos(2*${t})*${s}
          """))

    codegen = f.CodeGenerator(
            'MRAB',
            ode_component_type_map={
                "slow": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    ),
                "fast": f.ArrayType(
                    (1,),
                    f.BuiltinType('real (kind=8)'),
                    )
                },
            function_registry=freg,
            module_preamble="""
            ! lines copied to the start of the module, e.g. to say:
            ! use ModStuff
            """)

    code_str = codegen(code)

    # Build in conditionals to alter the timestep based on order such that all
    # tests pass

    if min_order == 3:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab_squarewave.f90", (
                read_file("test_mrab_squarewave.f90")
                .replace("MIN_ORDER", str(min_order - 0.3)+"d0")
                .replace("NUM_TRIPS_ONE", str(200))
                .replace("NUM_TRIPS_TWO", str(300)))),
            ],
            fortran_options=["-llapack", "-lblas"])
    elif min_order == 5:
        if method_name == 'Sqrs':
            # This is an especially problematic test in that it has errors that
            # are too low to achieve order convergence.
            run_fortran([
                ("abmethod.f90", code_str),
                ("test_mrab_squarewave.f90", (
                    read_file("test_mrab_squarewave.f90")
                    .replace("MIN_ORDER", str(min_order - 0.3)+"d0")
                    .replace("NUM_TRIPS_ONE", str(5))
                    .replace("NUM_TRIPS_TWO", str(10)))),
                ],
                fortran_options=["-llapack", "-lblas"])

        else:
            run_fortran([
                ("abmethod.f90", code_str),
                ("test_mrab_squarewave.f90", (
                    read_file("test_mrab_squarewave.f90")
                    .replace("MIN_ORDER", str(min_order - 0.3)+"d0")
                    .replace("NUM_TRIPS_ONE", str(130))
                    .replace("NUM_TRIPS_TWO", str(200)))),
                ],
                fortran_options=["-llapack", "-lblas"])
    else:
        run_fortran([
            ("abmethod.f90", code_str),
            ("test_mrab_squarewave.f90", (
                read_file("test_mrab_squarewave.f90")
                .replace("MIN_ORDER", str(min_order - 0.3)+"d0")
                .replace("NUM_TRIPS_ONE", str(100))
                .replace("NUM_TRIPS_TWO", str(150)))),
            ],
            fortran_options=["-llapack", "-lblas"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
