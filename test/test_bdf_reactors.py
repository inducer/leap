"""
Cantera example problem: reactor2.py

Two reactors connected with a piston, with heat loss to the environment

This script simulates the following situation. A closed cylinder with volume 2
m^3 is divided into two equal parts by a massless piston that moves with speed
proportional to the pressure difference between the two sides.  It is
initially held in place in the middle. One side is filled with 1000 K argon at
20 atm, and the other with a combustible 500 K methane/air mixture at 0.1 atm
(phi = 1.1). At t = 0 the piston is released and begins to move due to the
large pressure difference, compressing and heating the methane/air mixture,
which eventually explodes. At the same time, the argon cools as it expands.
The piston is adiabatic, but some heat is lost through the outer cylinder
walls to the environment.

Note that this simulation, being zero-dimensional, takes no account of shock
wave propagation. It is somewhat artifical, but nevertheless instructive.
"""

import numpy as np
import pytest


def test_vs_scipy():
    pytest.importorskip("scipy")
    pytest.importorskip("cantera")
    pytest.importorskip("pyrometheus")
    pytest.importorskip("jax")

    import scipy.linalg as la
    import scipy.integrate
    import cantera as ct
    from reactor_system import ReactorSystemOde
    import jax
    from jax import jit, jacfwd
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", 1)

    #------------------------------
    # First create each gas needed.
    #------------------------------

    # create an argon gas object and set its state
    ar = ct.Solution("argon.xml")
    ar.TP = 1000.0, 20.0 * ct.one_atm

    # use GRI-Mech 3.0 for the methane/air mixture, and set its initial state
    gas = ct.Solution("gri30.xml")
    gas.TP = 500.0, 0.2 * ct.one_atm
    gas.set_equivalence_ratio(1.1, "CH4:1.0", "O2:2, N2:7.52")

    ar_mass = ar.density
    gas_mass = gas.density

    # Now the problem is set up, and we're ready to solve it.
    #print('finished setup, begin solution...')

    n_steps = 300

    # Initialize reactor states for Leap.
    state = ar.T
    state = np.append(state, 1.0)
    state = np.append(state, ar.Y)
    state = np.append(state, gas.T)
    state = np.append(state, 1.0)
    state = np.append(state, gas.Y)

    # Modify mass fractions in the gas so that none are initially zero.
    sub_sum = 0
    for i in range(0, 53):
        if state[5+i] > 0:
            state[5+i] -= 1e-8
            sub_sum += 1e-8

    sub = sub_sum/50
    for i in range(0, 53):
        if state[5+i] == 0:
            state[5+i] += sub

    from leap.multistep import AdaptiveBDFMethodBuilder

    rtol = 1e-4
    atol = 1e-16

    method = AdaptiveBDFMethodBuilder("y", use_high_order=True, ndf=True,
                                      atol=atol, rtol=rtol, max_dt_growth=10,
                                      min_dt_shrinkage=0.2)

    code = method.generate()

    ode = ReactorSystemOde(ar, gas, np)
    ode_jax = ReactorSystemOde(ar, gas, jnp)

    #f = lambda y: ode_jax(t, y)
    def f(y):
        return ode_jax(t, y)

    jacobian = jit(jacfwd(f))

    def ReactorJac(t, y):
        return jacobian(y)

    def newton_solver(t, sub_y, coeff, guess):

        d = 0
        corr_norm = 1.0
        reltol = 1e-4
        abstol = 1e-16
        y_old = guess.copy()
        y_guess = guess.copy()
        corr_weights = np.zeros(58)
        # Match Scipy BDF
        newton_tol = 0.01
        newton_maxiter = 4
        corr_norm_old = None
        converged = False
        jac = ReactorJac(t, y_old)
        lu = la.lu_factor(np.eye(58) - coeff*jac, overwrite_a=True)
        # Check convergence w/weighted norm...
        for j in range(0, 58):
            corr_weights[j] = (reltol * np.abs(y_old[j]) + abstol)
        psi = -(sub_y - guess)
        for i in range(newton_maxiter):
            rhs = ode(t, y_guess)
            corr = la.lu_solve(lu, coeff*rhs - psi - d, overwrite_b=True)
            y_guess += corr
            d += corr
            # RMS norm:
            corr_norm = np.linalg.norm(corr / corr_weights) / 58 ** 0.5
            if corr_norm_old is not None:
                rate = corr_norm / corr_norm_old
            else:
                rate = None
            if rate is not None and rate / (1 - rate) * corr_norm < newton_tol:
                converged = True
                break
            corr_norm_old = corr_norm

        if converged is False:
            raise ValueError("Newton failed to converge")

        # Return RHS that gives y_n+1 = y_guess
        return (y_guess + psi - y_old)/coeff

    def solver_hook(solve_expr, solve_var, solver_id, guess):
        from dagrt.expression import match, substitute

        pieces = match("unk - <func>rhs(t=t, y=sub_y + coeff*unk)", solve_expr,
                        pre_match={"unk": solve_var})
        pieces["guess"] = guess
        return substitute("<func>solver(t, sub_y, coeff, guess)", pieces)

    from leap.implicit import replace_AssignImplicit
    code = replace_AssignImplicit(code, {"solve": solver_hook})

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")

    stepper_cls = codegen.get_class(code)

    t = 0.0
    # Match Scipy's initial timestep
    dt = 1.535031777168217e-05
    final_t = 4.e-4*n_steps

    stepper = stepper_cls(
            function_map={
                "<func>y": lambda t, y: ode(t, y),
                "<func>solver": newton_solver,
                })

    stepper.set_up(
            t_start=t, dt_start=dt,
            context={
                "y": state,
                })

    times = []
    values = []
    new_times = []
    new_values = []
    step_sizes = []
    last_t = 0.0
    istep = 0

    for event in stepper.run(t_end=final_t):
        if isinstance(event, stepper_cls.StateComputed):
            assert event.component_id == "y"
            # print("Reactor 2 Temp: T = ", event.state_component[3])
            new_times.append(event.t)
            new_values.append(event.state_component)
        elif isinstance(event, stepper_cls.StepCompleted):
            # print("Step completed: t = ", event.t)
            # print("Step completed: dt = ", event.t - last_t)
            istep += 1
            if not new_times:
                continue
            times.extend(new_times)
            values.extend(new_values)
            step_sizes.append(event.t - last_t)
            last_t = event.t
            # Get Leap to match Scipy's end behavior
            if times[-1] + stepper.dt > final_t:
                stepper.dt = final_t - times[-1]
            del new_times[:]
            del new_values[:]
        elif isinstance(event, stepper_cls.StepFailed):
            del new_times[:]
            del new_values[:]

    times = np.array(times)

    # Need to obtain the pressures and temperatures for both
    # the fast and slow reactors.

    # Slow reactor
    slow_temps = []
    slow_press = []
    slow_vol = []
    slow_energy = []
    for i in range(0, len(times)):
        ar.TDY = values[i][0], ar_mass/values[i][1], values[i][2]
        slow_temps.append(ar.T)
        slow_press.append(ar.P/1e5)
        slow_vol.append(values[i][1])
        slow_energy.append(ar.u)

    # Fast reactor
    fast_temps = []
    fast_press = []
    fast_vol = []
    fast_energy = []
    fast_neg = []
    for i in range(0, len(times)):
        gas.TDY = values[i][3], gas_mass/values[i][4], values[i][5:]
        fast_temps.append(gas.T)
        fast_press.append(gas.P/1e5)
        fast_vol.append(values[i][4])
        fast_energy.append(gas.u)
        fast_neg.append(0.0)
        for j in range(0, gas.n_species):
            # Track both magnitude and number of negative species
            if values[i][5+j] < 0:
                fast_neg[-1] -= values[i][5+j]

    step_sizes = np.array(step_sizes)

    # Now, replicate the process using Scipy and compare the two

    t = 0.0
    dt = 4.e-3
    final_t = 4.e-4*n_steps

    # New interface
    solver = scipy.integrate.BDF(ode, 0, state, final_t, rtol=rtol,
            atol=atol, jac=ReactorJac)

    # Need to obtain the pressures and temperatures for both
    # the fast and slow reactors.

    # Slow reactor
    sslow_temps = []
    sslow_press = []
    sslow_vol = []
    sslow_mass = []
    sslow_energy = []
    sfast_temps = []
    sfast_press = []
    sfast_vol = []
    sfast_mass = []
    sfast_energy = []
    sfast_species = []
    sfast_neg_sum = []
    sfast_neg_num = []
    for j in range(0, gas.n_species):
        sfast_species.append([])
    stimes = []
    ssteps = []

    # New interface
    while solver.status == "running" and solver.t < final_t:
        solver.step()
        ar.TDY = solver.y[0], ar_mass/solver.y[1], solver.y[2]
        sslow_vol.append(solver.y[1])
        sslow_temps.append(ar.T)
        sslow_mass.append(solver.y[1]*ar.density)
        sslow_energy.append(ar.u)
        sslow_press.append(ar.P/1e5)
        gas.TDY = solver.y[3], gas_mass/solver.y[4], solver.y[5:]
        frac_sum = 0
        num = 0
        for j in range(0, gas.n_species):
            sfast_species[j].append(solver.y[5+j])
            if solver.y[5+j] < 0:
                frac_sum -= solver.y[5+j]
                num += 1
        sfast_neg_sum.append(frac_sum)
        sfast_neg_num.append(num)
        sfast_vol.append(solver.y[4])
        sfast_temps.append(gas.T)
        sfast_press.append(gas.P/1e5)
        sfast_mass.append(solver.y[4]*gas.density)
        sfast_energy.append(gas.u)
        stimes.append(solver.t)
        ssteps.append(solver.t - solver.t_old)
        # print("Reactor 2 Temp: ", solver.y[3])
        # print("Time: ", solver.t)
        # print("Timestep: ", solver.t - solver.t_old)

    # Testing criterion
    # 1.) Ensure Leap solution doesn't contain negative mass fractions
    assert max(fast_neg) < 1e-16
    # 2.) Ensure final Reactor 2 temperatures of Scipy and Leap are the same
    assert abs(fast_temps[-1] - sfast_temps[-1]) < 1e-1
    # 3.) As a proxy for similar (but not exact) adaptive timestepping,
    #     ensure that Scipy and Leap took a similar number of steps
    #     to reach the same end time.
    assert abs((len(times) - len(stimes))/len(stimes)) < 0.05


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
