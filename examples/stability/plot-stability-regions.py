from __future__ import division, print_function


def plot_stability_region(code, parallel=None, scale_factor=None, **kwargs):
    from leap.stability import find_stability_region
    points = find_stability_region(code, parallel=parallel,
            prec=1e-2, n_angles=100, origin=-.3)

    if scale_factor is not None:
        points = points * scale_factor

    from matplotlib.pyplot import fill
    fill(points.real, points.imag, **kwargs)


def main(save_pdfs=True):
    import matplotlib.pyplot as pt
    pt.rc("font", size=15)
    #title("Stability Region")
    pt.xlabel(r"Re $\lambda$ / RHS calls")
    pt.ylabel(r"Im $\lambda$ / RHS calls")
    pt.grid()

    import leap.rk as rk  # noqa
    import leap.multistep as multistep  # noqa

    for label, method, factor in [
            #("ode23", rk.ODE23Method("y", use_high_order=True), 1),
            #("ab2", multistep.AdamsBashforthMethod("y", 2), 1),
            #("ab2", multistep.AdamsBashforthMethod("y", 2, static_dt=True), 1),
            ("ab3", multistep.AdamsBashforthMethod("y", 3, static_dt=True), 1),
            ("ab4-trig-0.7", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 0.7),
                static_dt=True), 1),
            ("ab4-trig-0.9", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 0.9),
                static_dt=True), 1),
            ("ab4-trig-1.1", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 1.1),
                static_dt=True), 1),
            ("ab4-trig-1.3", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 1.3),
                static_dt=True), 1),
            ("ab4-trig-1.5", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 1.5),
                static_dt=True), 1),
            ("ab4-trig-2", multistep.AdamsBashforthMethod(
                "y", multistep.ABTrigMonomialIntegrationFunctionFamily(4, 2),
                static_dt=True), 1),
            #("ab3-4", multistep.AdamsBashforthMethod("y", 3, hist_length=4), 1),
            #("ab3-4", multistep.AdamsBashforthMethod("y", 3, hist_length=5), 1),
            #("ab3-4", multistep.AdamsBashforthMethod("y", 3, hist_length=6), 1),
            #("ab4", multistep.AdamsBashforthMethod("y", 4, static_dt=True), 1),
            #("lserk", rk.LSRK4Method("y"), 1/5),
            #("rk4", rk.RK4Method("y"), 1/4),
            ]:

        code = method.generate()
        plot_stability_region(code, label=label, alpha=0.3, scale_factor=factor,
                parallel=True)

    pt.legend(labelspacing=0.1, borderpad=0.3, loc="best")
    if save_pdfs:
        pt.tight_layout()
        pt.savefig("stab-regions.pdf")
        xmin, xmax = pt.ylim()
        xsize = xmax-xmin

        pt.gca().set_aspect("equal")
        pt.ylim([-xsize/2*0.75, xsize/2*0.75])
        pt.tight_layout()
        pt.savefig("stab-regions-eq-aspect.pdf")

    else:
        pt.show()


if __name__ == "__main__":
    main()
