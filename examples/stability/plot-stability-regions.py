from __future__ import division, print_function


def plot_stability_region(code, parallel=None, scale_factor=None, **kwargs):
    from leap.stability import find_stability_region
    points = find_stability_region(code, parallel=parallel,
            prec=1e-2, n_angles=100, origin=-.3)

    if scale_factor is not None:
        points = points * scale_factor

    from matplotlib.pyplot import fill
    fill(points.real, points.imag, **kwargs)


def main(save_pdfs=False):
    import matplotlib.pyplot as pt
    pt.rc("font", size=20)
    #title("Stability Region")
    pt.xlabel(r"Re $\lambda$")
    pt.ylabel(r"Im $\lambda$")
    pt.grid()

    import leap.rk as rk
    import leap.ab as ab

    for label, method, factor in [
            #("ode23", rk.ODE23Method("y", use_high_order=True), 1),
            #("ab2", ab.AdamsBashforthMethod("y", 2), 1),
            ("5*ab3", ab.AdamsBashforthMethod("y", 3), 5),
            #("ab4", ab.AdamsBashforthMethod("y", 4), 1),
            ("lserk", rk.LSRK4Method("y"), 1),
            ("rk4", rk.RK4Method("y"), 1),
            ]:

        code = method.generate()
        plot_stability_region(code, label=label, alpha=0.3, scale_factor=factor)

    pt.legend(labelspacing=0.1, borderpad=0.3, loc="best")
    if save_pdfs:
        pt.savefig("stab-regions.pdf")
        xmin, xmax = pt.xlim()
        xsize = xmax-xmin

        pt.gca().set_aspect("equal")
        pt.ylim([-xsize/2*0.75, xsize/2*0.75])
        pt.savefig("stab-regions-eq-aspect.pdf")

    else:
        pt.show()


if __name__ == "__main__":
    main()
