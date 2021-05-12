#!/usr/bin/env python
"""Solves the 1D wave equation

  u_tt - c(x)^2 u_xx = 0
  u_t(0) = init'
  u(0) = init

with piecewise constant coefficients c(x) using a multirate multistep method.
"""


import argparse
import fnmatch
import logging
import matplotlib
import numpy as np
import numpy.linalg as la
import os


from contextlib import contextmanager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAPER_OUTPUT = int(os.environ.get("PAPER_OUTPUT", 0))
OUT_DIR = os.environ.get("OUT_DIR", ".")


# {{{ matplotlib setup

if PAPER_OUTPUT:
    matplotlib.use("pgf")
else:
    matplotlib.use("agg")

import matplotlib.pyplot as plt  # noqa


FONTSIZE = 9
LINEWIDTH = 0.5


def initialize_matplotlib():
    plt.rc("font", family="serif")
    plt.rc("text", usetex=False)
    plt.rc("xtick", labelsize=FONTSIZE)
    plt.rc("ytick", labelsize=FONTSIZE)
    plt.rc("axes", labelsize=FONTSIZE)
    plt.rc("axes", titlesize=FONTSIZE)
    plt.rc("axes", linewidth=LINEWIDTH)
    plt.rc("pgf", rcfonts=True)
    plt.rc("lines", linewidth=LINEWIDTH)


if PAPER_OUTPUT:
    initialize_matplotlib()

# }}}


# {{{ problem discretization

class VariableCoeffWaveEquationProblem:
    """Discretizes an instance of the 1D variable coefficient wave equation
    problem, providing initial values and right-hand side evaluations.

    This turns the problem instance into the 1D system

         u_t = v
         v_t = c(x)^2 * u_xx.

    The system state is separated into components, i.e. regions in which c(x)
    is constant. Components are equispaced in the interval (0, 1) and
    correspond to the number of coefficients passed to the constructor.

    The discretization uses a centered difference stencil.

    The interface conditions to the continuous problem require continuity of u
    and c(x)^2 * u_x at the interfaces.  These conditions are imposed on the
    discretization using a second-order biased stencil by ensuring that the
    values at the interface implied by the stencil, when applied from both the
    left and the right sides, are the same. For more on interface conditions,
    see

        David L. Brown.
        A note on the numerical solution of the wave equation with piecewise
            smooth coefficients.
        Math. Comp., 42(166):369–391, 1984.
        doi: 10.2307/2007591.

    """

    def __init__(self, ngridpoints, component_coeffs, decay=0, diffusion=0):
        """
        Args:
            ngridpoints: Number of mesh points
            component_coeffs: Coefficients of components
                (=squares of wave speeds)
        """

        ncomponents = len(component_coeffs)

        grid = np.linspace(0, 1, ngridpoints + 1, endpoint=False)
        component_times = np.r_[grid[::1+ngridpoints//ncomponents], 1]

        # Left endpoint of grid is fixed at 0.
        self.grid = grid[1:]
        self.h = grid[1] - grid[0]
        self.ncomponents = ncomponents

        # n = Number of simulation unknowns
        #
        # Interface points (those between adjacent components) are not
        # included: the solution value at the interface points is inferred
        # using the stencil.
        self.n = 2 * (ngridpoints - self.ncomponents + 1)

        component_indices = []
        component_sizes = []

        for start, stop in zip(component_times, component_times[1:]):
            indices = (start < grid) & (grid < stop)
            component_indices.append(indices)
            component_sizes.append(2 * sum(indices))

        assert sum(component_sizes) == self.n

        self.component_indices = component_indices
        self.component_sizes = component_sizes
        self.coeffs = component_coeffs

        self.decay = decay
        self.diffusion = diffusion

        # Initial values
        f = np.sin(2 * 6 * np.pi * grid)
        f_prime = np.zeros_like(f)

        # Initial conditions
        self.u0_by_component = []

        for index in component_indices:
            self.u0_by_component.append(np.hstack([f[index], f_prime[index]]))

        self.u0 = np.hstack(self.u0_by_component)

    @property
    def dt_ref(self):
        return (1/2) * self.h / min(self.coeffs)

    def split_view(self, comp):
        return comp[:len(comp) // 2], comp[len(comp) // 2:]

    def rhs(self, from_component_num, to_component_num, t, **kwargs):
        """Evaluate the right-hand side for the given component interaction.

        Args:
            from_component_num: Source component
            to_component_num: Target component
            kwargs: Component values by name. "compN" is the N-th component
        """
        component = kwargs["comp%d" % from_component_num]
        old_u, old_ut = self.split_view(component)

        result = np.zeros(
                self.component_sizes[to_component_num],
                dtype=component.dtype)
        res_ut, res_utt = self.split_view(result)

        # Contribution from left panel
        if from_component_num == to_component_num - 1:
            # Interface conditions
            r = self.coeffs[from_component_num] / (3 * (
                    self.coeffs[from_component_num]
                    + self.coeffs[to_component_num]))
            u_l = r * (4 * old_u[-1] - old_u[-2])

            res_ut[0] = self.diffusion * u_l / self.h ** 2
            res_utt[0] = self.coeffs[to_component_num] * u_l / self.h ** 2

        # Contribution from self panel
        elif from_component_num == to_component_num:
            old_uxx_mid = (
                    (old_u[:-2] - 2 * old_u[1:-1] + old_u[2:])
                    / self.h ** 2)
            old_uxx_l = (-2 * old_u[0] + old_u[1]) / self.h ** 2
            old_uxx_r = (old_u[-2] - 2 * old_u[-1]) / self.h ** 2

            if from_component_num > 0:
                # Left interface condition
                r = self.coeffs[from_component_num] / (3 * (
                        self.coeffs[from_component_num - 1]
                        + self.coeffs[from_component_num]))

                old_u_l = r * (4 * old_u[0] - old_u[1])
                old_uxx_l += old_u_l / self.h ** 2

            if from_component_num < self.ncomponents - 1:
                # Right interface condition
                r = self.coeffs[from_component_num] / (3 * (
                        self.coeffs[from_component_num + 1]
                        + self.coeffs[from_component_num]))

                old_u_r = r * (4 * old_u[-1] - old_u[-2])
                old_uxx_r += old_u_r / self.h ** 2

            old_uxx = np.hstack([[old_uxx_l], old_uxx_mid, [old_uxx_r]])
            res_ut[:] = old_ut - self.decay * old_u + self.diffusion * old_uxx
            res_utt[:] = self.coeffs[from_component_num] * old_uxx

        # Contribution from right panel
        elif from_component_num == to_component_num + 1:
            # Interface condition
            r = self.coeffs[from_component_num] / (3 * (
                    self.coeffs[from_component_num]
                    + self.coeffs[to_component_num]))
            u_r = r * (4 * old_u[0] - old_u[1])

            res_ut[-1] = self.diffusion * u_r / self.h ** 2
            res_utt[-1] = self.coeffs[to_component_num] * u_r / self.h ** 2

        else:
            raise ValueError("Components not spatially adjacent")

        return result

    def full_solution(self, components):
        """Reconstruct a full solution of the problem from a list of solution
        components.

        """
        result = []

        for i in range(self.ncomponents):
            u = self.split_view(components[i])[0]
            result.append(u)

            if i < self.ncomponents - 1:
                u_next, _ = self.split_view(components[i + 1])
                # Reconstruct midpoint from interface condition.
                r = 1 / (3 * (
                        self.coeffs[i] + self.coeffs[i + 1]))
                u_l = self.coeffs[i] * r * (4 * u[-1] - u[-2])
                u_r = self.coeffs[i+1] * r * (4 * u_next[0] - u_next[1])
                result.append([u_l + u_r])

        result = np.hstack(result)
        return result

# }}}


# {{{ multirate method generator

def make_3_component_multirate_method(
            problem, ratios, order=3, code_only=False, return_rhs_map=False):
    """Return the object that drives the multirate method for the given
    parameters.

    """
    from leap.multistep.multirate import (
            MultiRateMultiStepMethodBuilder, MultiRateHistory)

    ncomponents = 3
    assert problem.ncomponents == ncomponents

    code = MultiRateMultiStepMethodBuilder(
            default_order=3,
            system_description=(
                ("dt", "comp0",
                 "=",
                 MultiRateHistory(ratios[0], "<func>rhs0to0", ("comp0",)),
                 MultiRateHistory(ratios[0], "<func>rhs1to0", ("comp1",))),
                ("dt", "comp1",
                 "=",
                 MultiRateHistory(ratios[1], "<func>rhs0to1", ("comp0",)),
                 MultiRateHistory(ratios[1], "<func>rhs1to1", ("comp1",)),
                 MultiRateHistory(ratios[1], "<func>rhs2to1", ("comp2",))),
                ("dt", "comp2",
                 "=",
                 MultiRateHistory(ratios[2], "<func>rhs1to2", ("comp1",)),
                 MultiRateHistory(ratios[2], "<func>rhs2to2", ("comp2",)))),
            static_dt=True).generate()

    from functools import partial

    if code_only and not return_rhs_map:
        return code

    rhs_map = {}
    for i in range(3):
        rhs_map["<func>rhs%dto%d" % (i, i)] = partial(problem.rhs, i, i)
        if i > 0:
            rhs_map["<func>rhs%dto%d" % (i, i - 1)] = (
                    partial(problem.rhs, i, i - 1))
        if i < ncomponents - 1:
            rhs_map["<func>rhs%dto%d" % (i, i + 1)] = (
                    partial(problem.rhs, i, i + 1))

    if code_only:
        if return_rhs_map:
            return code, rhs_map
        else:
            return code

    from dagrt.codegen import PythonCodeGenerator
    MRABMethod = PythonCodeGenerator(class_name="MRABMethod").get_class(code)  # noqa

    if return_rhs_map:
        return MRABMethod(rhs_map), rhs_map
    else:
        return MRABMethod(rhs_map)

# }}}


# {{{ example plotter

def plot_example(ngridpoints):
    problem = VariableCoeffWaveEquationProblem(
            ngridpoints, component_coeffs=[16, 4, 1])
    stepper = make_3_component_multirate_method(problem, ratios=(1, 1, 1))

    dt = 0.01 * problem.dt_ref
    t_end = 0.5

    # {{{ step solution

    stepper.set_up(
            t_start=0,
            dt_start=dt,
            context={
                "comp0": problem.u0_by_component[0],
                "comp1": problem.u0_by_component[1],
                "comp2": problem.u0_by_component[2]})

    vals_by_component = {}
    times_by_component = {}

    for event in stepper.run(t_end=t_end):
        if isinstance(event, stepper.StateComputed):
            vals_by_component.setdefault(
                    event.component_id, []).append(event.state_component)
            times_by_component.setdefault(
                    event.component_id, []).append(event.t)

    n = len(vals_by_component["comp0"])
    vals = []
    for i in range(n):
        soln = problem.full_solution([
                vals_by_component[comp][i]
                for comp in ("comp0", "comp1", "comp2")])
        # Add Dirichlet BCs for proper axis labeling
        vals.append(np.r_[0, soln, 0])

    vals = np.vstack(vals)

    # }}}

    figure = plt.figure(figsize=(4, 2), dpi=300)
    axis = figure.add_subplot(111)

    image = axis.imshow(
            vals,
            cmap="Spectral",
            aspect="auto",
            origin="lower",
            extent=(0, 1, 0, 0.5),
            vmin=-1.5,
            vmax=1.5)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    plt.colorbar(image, cax=cax)

    """
    axis.plot_surface(
            problem.grid.reshape(-1, 1),
            np.array(times_by_component["comp0"]).reshape(1, -1),
            vals.T,
            rcount=100,
            ccount=100,
            cmap="Spectral")
    """

    axis.set_xlabel("$x$")
    axis.set_ylabel("$t$")
    axis.set_aspect(1)
    axis.set_xticks([0, 1/3, 2/3, 1])
    axis.set_xticklabels(["0", "1/3", "2/3", "1"])
    axis.set_title("Solution to 1D Wave Equation with Variable Coefficients")

    suffix = "pgf" if PAPER_OUTPUT else "pdf"
    filename = os.path.join(OUT_DIR, "wave-problem.%s" % suffix)

    plt.savefig(filename, bbox_inches="tight")
    logger.info("wrote to '%s'" % filename)

# }}}


# {{{ stability experiment

@contextmanager
def timer(name):
    from time import time
    logging.info(f"start: {name}")
    start = time()
    yield
    end = time()
    logging.info("finished: {name}: {time} seconds".format(
        name=name, time=end - start))


def generate_mrab_step_matrix(ngridpoints, coeffs, substep_ratio, filename):
    problem = VariableCoeffWaveEquationProblem(ngridpoints, coeffs)

    code, rhs_map = make_3_component_multirate_method(
            problem, substep_ratio, code_only=True, return_rhs_map=True)
    from leap.step_matrix import StepMatrixFinder

    finder = StepMatrixFinder(
            code,
            function_map=rhs_map,
            exclude_variables=["<p>bootstrap_step"])

    with timer(f"Constructing MRAB({substep_ratio}) matrix"):
        component_sizes = {}
        for var in finder.variables:
            for i in range(problem.ncomponents):
                if "comp%d" % i in var:
                    component_sizes[var] = problem.component_sizes[i]
                    break
            else:
                raise ValueError("cannot infer size of variable: %s" % var)

        import pprint
        pprint.pprint(component_sizes)

        mat = finder.get_phase_step_matrix(
                "primary",
                shapes=component_sizes,
                sparse=True)

        with open(filename, "wb") as outf:
            import pickle
            pickle.dump(mat, outf)

    logging.info("%s: %d nnz, size %s", filename, len(mat.data), mat.shape)


def compute_all_stable_timesteps(filenames, stable_dts_outf):
    import re

    rows = [["Intervals", r"Stable $\Delta t$",
             r"$\Delta t / \Delta t_{(1,1,1)}$"]]

    first = True

    for fname in filenames:
        import pickle

        with timer(f"loading {fname}"):
            with open(fname, "rb") as infile:
                mat = pickle.load(infile)

        logging.info("Computing stable timestep")
        dt = compute_stable_timestep(mat)

        if first:
            dt_1 = dt

        intervals = tuple(
                int(i) for i in
                re.match(
                    r"mat\d+-(\d+)-(\d+)-(\d+)\.pkl",
                    os.path.basename(fname)).groups())

        row = [str(intervals), "%.2e" % dt]

        if first:
            row.append("---")
        else:
            ratio = dt / dt_1
            row.append("%.1f" % ratio)

        first = False
        rows.append(row)

    print(tabulate(rows, col_fmt="cSc"), file=stable_dts_outf)

    if hasattr(stable_dts_outf, "name"):
        logger.info("Wrote '%s'", stable_dts_outf.name)


def compute_stable_timestep(step_matrix, tol=0, prec=1e-15):
    def as_dense(mat):
        from scipy.sparse import coo_matrix
        indices = np.asarray(mat.indices)
        return coo_matrix(
            (mat.data, (indices[:, 0], indices[:, 1])),
            shape=mat.shape).todense()

    from leap.step_matrix import fast_evaluator
    evaluate_mat = fast_evaluator(step_matrix, sparse=True)

    def spectral_radius(dt):
        mat = as_dense(evaluate_mat({"<dt>": dt}))
        eigvals = la.eigvals(mat)
        radius = np.max(np.abs(eigvals))
        logging.info("%s -> spectral radius %s", dt, radius)
        return radius

    def is_stable(dt):
        return spectral_radius(dt) <= 1

    from leap.stability import find_truth_bdry
    return find_truth_bdry(is_stable, prec=1e-7, start_magnitude=1e-4)

# }}}


# {{{ accuracy experiment

def multirate_accuracy_experiment(errors_outf):
    results_by_substep_ratio = {}
    eocs = []

    substep_ratios = [(1, 1, 1), (1, 2, 2), (1, 2, 4)]
    dts_orig = 2 ** np.arange(-11, -17., -1)
    problem = VariableCoeffWaveEquationProblem(100, (16, 4, 1))
    component_names = ("comp0", "comp1", "comp2")

    initial_context = {
            name: problem.u0_by_component[i]
            for i, name in enumerate(component_names)}

    for substep_ratio in substep_ratios:
        dts = dts_orig * max(substep_ratio)

        for dt in dts:
            logger.info("running ratio '%s' with dt=%e", substep_ratio, dt)
            stepper = make_3_component_multirate_method(
                    problem,
                    ratios=substep_ratio)
            stepper.set_up(t_start=0, dt_start=dt, context=initial_context)

            state_components = {}

            for event in stepper.run(t_end=0.5):
                if isinstance(event, stepper.StateComputed):
                    state_components[event.component_id] = \
                            event.state_component

            results_by_substep_ratio.setdefault(substep_ratio, []).append(
                    problem.full_solution(list(
                        state_components[component]
                        for component in component_names)))

    rows = []
    rows.append(
            [r"$\Delta t_\text{fast}$"]
            + [str(substep) for substep in substep_ratios])

    from pytools.convergence import EOCRecorder
    eocs = {s: EOCRecorder() for s in substep_ratios}

    for i_dt, dt in enumerate(dts_orig[:-1]):
        row = ["%.2e" % dt]
        for substep_ratio in substep_ratios:
            ref_result = results_by_substep_ratio[substep_ratio][-1]
            result = results_by_substep_ratio[substep_ratio][i_dt]
            error = la.norm(result - ref_result, ord=np.inf)
            row.append("%.2e" % error)
            eocs[substep_ratio].add_data_point(dt, error)
        rows.append(row)

    if PAPER_OUTPUT:
        rows.append([r"\midrule"])
        row = [r"\multicolumn{1}{l}{Order}"]
    else:
        row = ["Order"]

    for s in substep_ratios:
        row.append("%.2f" % eocs[s].order_estimate())
    rows.append(row)

    print(
            tabulate(rows, col_fmt="S" * (1 + len(substep_ratios))),
            file=errors_outf)

    if hasattr(errors_outf, "name"):
        logger.info("Wrote '%s'", errors_outf.name)

# }}}


# {{{ table generation

def tabulate_latex(rows, col_fmt):
    result = []
    result.append(r"\begin{tabular}{%s}" % col_fmt)
    result.append(r"\toprule")
    result.append(
        " & ".join((r"\multicolumn{1}{c}{%s}" % t) for t in rows[0]) + r"\\")
    result.append(r"\midrule")
    for row in rows[1:]:
        result.append(" & ".join(row) + r"\\")
    result.append(r"\bottomrule")
    result.append(r"\end{tabular}")
    return "\n".join(result)


def tabulate_ascii(rows, col_fmt):
    del col_fmt
    from pytools import Table
    result = Table()
    for row in rows:
        result.add_row(row)
    return str(result)

# }}}


if PAPER_OUTPUT:
    tabulate = tabulate_latex
else:
    tabulate = tabulate_ascii


def open_or_stdout(filename):
    if not PAPER_OUTPUT:
        import sys
        return sys.stdout
    else:
        return open(os.path.join(OUT_DIR, filename), "w")


# {{{ experimental drivers

def run_plot_experiment():
    # Plot an example solution.
    plot_example(1000)


def run_accuracy_experiment():
    errors_outf = open_or_stdout("mrab-errors.tex")
    multirate_accuracy_experiment(errors_outf)


def run_stability_experiment():
    # Generate stability results.
    # Generating the step matrices takes a long time.
    step_ratios = (
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 4),
            (1, 2, 6),
    )

    filenames = []

    ngridpoints = 100

    for step_ratio in step_ratios:
        filename = os.path.join(
                OUT_DIR,
                "mat%d-%d-%d-%d.pkl" % ((ngridpoints,) + step_ratio))

        if not os.path.exists(filename):
            generate_mrab_step_matrix(
                    ngridpoints, (16, 4, 1), step_ratio, filename)
        else:
            logger.info("using saved step matrix '%s'" % filename)

        filenames.append(filename)

    stable_dts_outf = open_or_stdout("mrab-stable-dts.tex")
    compute_all_stable_timesteps(filenames, stable_dts_outf)

# }}}


EXPERIMENTS = ("plot", "accuracy", "stability")


def parse_args():
    names = ["  - '%s'" % name for name in EXPERIMENTS]
    epilog = "\n".join(["experiment names:"] + names)
    epilog += "\n".join([
            "\n\nenvironment variables:",
            "   - OUT_DIR: output path (default '.')",
            "   - PAPER_OUTPUT: if set to true, generate paperable outputs"])

    parser = argparse.ArgumentParser(
            description="Runs one or more experiments.",
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
            "-x",
            metavar="experiment-name",
            action="append",
            dest="experiments",
            default=[],
            help="Adds an experiment to the list to be run "
                 "(accepts wildcards) (may be given multiple times)")

    parser.add_argument(
            "--all",
            action="store_true",
            dest="run_all",
            help="Runs all experiments")

    parser.add_argument(
            "--except",
            action="append",
            metavar="experiment-name",
            dest="run_except",
            default=[],
            help="Removes an experiment from the list to be run "
                 "(accepts wildcards) (may be given multiple times)")

    parse_result = parser.parse_args()

    result = set()

    if parse_result.run_all:
        result = set(EXPERIMENTS)

    for experiment in EXPERIMENTS:
        for pat in parse_result.experiments:
            if fnmatch.fnmatch(experiment, pat):
                result.add(experiment)
                continue

    to_discard = set()
    for experiment in EXPERIMENTS:
        for pat in parse_result.run_except:
            if fnmatch.fnmatch(experiment, pat):
                to_discard.add(experiment)
                continue
    result -= to_discard

    return result


def main():
    experiments = parse_args()

    if "plot" in experiments:
        run_plot_experiment()

    if "accuracy" in experiments:
        run_accuracy_experiment()

    if "stability" in experiments:
        run_stability_experiment()


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
