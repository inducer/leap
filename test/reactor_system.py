import cantera as ct
import pyrometheus as pyro


class ReactorSystemOde(object):
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
    def __init__(self, gas1, gas2, np):
        self.gas1 = pyro.get_thermochem_class(gas1)(usr_np=np)
        self.gas2 = pyro.get_thermochem_class(gas2)(usr_np=np)
        self.gas2_ct = gas2
        self.env = ct.Reservoir(ct.Solution('air.xml'))
        # Initial volume of each reactor is 1.0, so...
        self.gas1_mass = gas1.density
        self.gas2_mass = gas2.density
        self.gas1_mw = gas1.molecular_weights
        self.gas2_mw = gas2.molecular_weights
        self.np = np

    def __call__(self, t, y):

        # State vector is [U1, V1, Y_1, Y_2, ... Y_K, U2, V2, Y_1, .... Y_K]
        # Set gases.
        temp1 = y[0]
        rho1 = self.gas1_mass * (1.0/y[1])
        p1 = self.gas1.get_pressure(rho1, temp1, self.np.asarray([y[2]]))

        temp2 = y[3]
        rho2 = self.gas2_mass * (1.0/y[4])
        p2 = self.gas2.get_pressure(rho2, temp2, self.np.asarray(y[5:]))

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        area = 1.0
        k = 0.5e-4
        u1 = 100.0
        dvdt_1 = k*area*(p1 - p2)

        # Mass fraction rate of change (via production
        # rates as is typical)
        # Reactor 1 is pure argon
        wdot_1 = 0.0
        dydt_1 = wdot_1 * self.gas1_mw * (1.0 / rho1)

        # Internal energy rate of change
        # Pressure work first.
        dtempdt_1 = -p1*dvdt_1
        # Include heat transfer via the piston wall
        dtempdt_1 += -area*u1*(temp1 - temp2)
        # Partial molar internal energies
        e0_rt1 = (self.gas1.get_species_enthalpies_rt(temp1) - 1.0) * \
            self.gas1.gas_constant * temp1
        dtempdt_1 += -self.np.dot(e0_rt1, wdot_1*(1.0/rho1)*self.gas1_mass)
        dtempdt_1 = dtempdt_1 * (1.0
                / (self.gas1_mass
                    * self.gas1.get_mixture_specific_heat_cv_mass(temp1,
                        self.np.asarray([y[2]]))))

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        dvdt_2 = k*area*(p2 - p1)

        # Mass fraction rate of change (via production
        # rates as is typical)
        wdot_2 = self.gas2.get_net_production_rates(rho2, temp2, y[5:])
        dydt_2 = wdot_2 * self.gas2_mw * (1.0 / rho2)

        # Internal energy rate of change
        # Pressure work first.
        dtempdt_2 = -p2*dvdt_2
        # Include heat transfer via the piston wall
        dtempdt_2 += -area*u1*(temp2 - temp1)
        # Include heat loss to the environment (air reservoir)
        # via specified wall
        area2 = 1.0
        u2 = 500.0
        dtempdt_2 += -area2*u2*(temp2 - self.env.T)
        e0_rt2 = (self.gas2.get_species_enthalpies_rt(temp2) - 1.0) \
            * self.gas2.gas_constant * temp2
        dtempdt_2 += -self.np.dot(e0_rt2, wdot_2*(1.0/rho2)*self.gas2_mass)
        dtempdt_2 = dtempdt_2 * (1.0
                / (self.gas2_mass
                    * self.gas2.get_mixture_specific_heat_cv_mass(temp2,
                        self.np.asarray(y[5:]))))

        return self.np.hstack((self.np.hstack((dtempdt_1, dvdt_1, dydt_1)),
                          self.np.hstack((dtempdt_2, dvdt_2, dydt_2))))
