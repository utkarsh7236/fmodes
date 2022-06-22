#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

import cowling_approximation.initialize as initialize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import ode
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt


class CowlingApproximation:
    """
    Cowling Appriximation Class for fmode calculations. In the future this can be extended to the generalized GR
    version.
    """

    def __init__(self):
        """
        Initialize parameters being used in integration as NoneType. Specify some parameters types. Set constants,
        data management system and default maximum number of integrations.
        """
        self.e, self.p, self.EOS, self.path, self.dedp, self.pbar, self.v0 = None, None, None, None, None, None, None
        self.W0, self.U0, self.init_VEC, self.p_max, self.p_min, self.p_max = None, None, None, None, None, None
        self.p0, self.e0, self.p_c, self.e_c, self.m0, self.omega, self.l = None, None, None, None, None, None, None
        self.r_i, self.m, self.v, self.w, self.u, self.r_arr, self.e_arr = None, None, None, None, None, None, None
        self.p_arr, self.m_R, self.r_R, self.f, self.res, self.omega_arr = None, None, None, None, None, None
        self.loss_arr = None
        self.omega_vals, self.loss_vals = [], []

        # Maximum number of integrations
        self.n_iter_max = 20000

        # Set constants and data management system
        self.const = initialize.Constants()
        self.data = initialize.DataManagement()

    @staticmethod
    def _get_ep(e, p):
        """
        Returns Equation of State (EOS) interpolated from the tabular values given.
        :param e: Energy density in units of CGS
        :param p: Pressure in units of CGS
        :return: Smoothing functional which interpolates pressure, energy density values
        """
        f_e_smooth = interp1d(p, e, fill_value="extrapolate", kind="cubic")
        return f_e_smooth

    @staticmethod
    def _get_pe(p, e):
        """
        Returns inverted Equation of State (EOS) interpolated from the tabular values given.
        :param e: Energy density in units of CGS
        :param p: Pressure in units of CGS
        :return: Smoothing functional which interpolates pressure, energy density values in an inverted format.
        """
        f_e_smooth = interp1d(e, p, fill_value=(0, 0), kind="cubic", bounds_error=True)
        return f_e_smooth

    def read_data(self, path):
        """
        Reads Equation of State (EOS) data from a given txt file into readable e, p and EOS arrays.
        :param path: Path to equation of state in tabular format.
        :return: None
        """
        self.path = path
        df = pd.read_csv(path)
        e, p = self.data.df_to_ep(df)
        EOS = self._get_ep(e, p)
        self.e_arr = e
        self.p_arr = p
        self.EOS = EOS
        return None

    def load_dedp(self):
        """
        Transition function which loads gradient in functional form to be used in TOV integration.
        :return: None
        """
        self.dedp = self._dedP(self.p_arr, self.e_arr)
        return None

    @staticmethod
    def _dedP_helper(p_arr, e_arr):
        """
        Computes gradient between pressure and energy density.
        :param p_arr: Pressure array in CGS
        :param e_arr: Energy density array in CGS
        :return: de/dP gradient along with the energy density array used.
        """
        return np.gradient(e_arr, p_arr), e_arr

    def _dedP(self, p_arr, e_arr):
        """
        Compiler function for computing energy density to be used in TOV integration. This equation uses helper
        functions to generate a functional form which predicts a single de/dp given the e, p tabular arrays.
        :param p_arr: Pressure array in CGS
        :param e_arr: Energy density array in CGS
        :return: Functional de/dP as a function of e which predicts at a single point of e
        """
        dedp_helper, e_arr = self._dedP_helper(p_arr, e_arr)
        return interp1d(e_arr, dedp_helper, fill_value="extrapolate", kind="cubic")

    def _drhodP(self, e):
        """
        Convert de/dP into readble CGS units by dividing by c^2
        :param e: Energy density array in CGS
        :return: drho/dP which has units of inverse sound speed
        """
        c = self.const.c
        return (c ** -2) * self.dedp(e)

    def _lamda_metric(self, M, R):
        """
        Compute lamda element of generalized metric by matching to swarzchild metric. Here exp(lamda) = schild.
        :param M: Enclosed Mass
        :param R: Radius of encolosed mass
        :return: Lamda value given the swarzchild metric.
        """
        G = self.const.G
        c = self.const.c
        return -np.log((1 - 2 * G * M / (c ** 2 * R)))

    def _dMdr(self, r, e):
        """
        First order mass differential equation (TOV Equation 1)
        :param r: Radius at given point
        :param e: Energy density at given radius/pressure
        :return: Rate of change of mass/radius
        """
        c = self.const.c
        return 4 * np.pi * r ** 2 * (e / (c ** 2))

    def _b(self, r, M):
        """
        Functional of swarchild as defined in arXiv 2205.02081, ignored factor of 2 due to their typo.
        :param r: Radius at given point
        :param M: Enclosed mass
        :return: Functional value of mass/radius dependent term in swarzchild.
        """
        G = self.const.G
        c = self.const.c
        return (G * M) / ((c ** 2) * r)

    @staticmethod
    def _dvdr(r, Q, lamda):
        """
        Change in metric element nu as integration continues through the star.
        :param r: Radius at given point
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param lamda: Metric value for a given mass/radius.
        :return: Temporal metric element.
        """
        return (2 / r) * np.exp(lamda) * Q

    def _Q(self, r, P, M):
        """
        Transition function, dependent on the TOV equations.
        :param r: Radius at given point
        :param P: Pressure at a given radius.
        :param M: Enclosed mass
        :return: Functional value of TOV transition function as determined by arXiv 2205.02081.
        """
        G = self.const.G
        c = self.const.c
        frac = (4 * np.pi * G * (r ** 2) * P) / (c ** 4)
        return self._b(r, M) + frac

    def _dPdr(self, r, P, M, e):
        """
        First order pressure differential equation (TOV Equation 2)
        :param r: Radius at given point
        :param P: Pressure at a given radius.
        :param M: Enclosed mass
        :param e: Energy density at a given pressure
        :return: Rate of change of pressure/radius
        """
        G = self.const.G
        c = self.const.c
        num = (M + 4 * np.pi * (r ** 3) * P / (c ** 2))
        dem = r * (r - 2 * G * M / (c ** 2))
        return -1 * (e + P) * G / (c ** 2) * num / dem

    def _dWdlnr(self, r, W, U, Q, lamda, l, omega, v, c_ad2_inv):
        """
        First order differential equation for W term in coupled equations (Cowling Approximation 1)
        :param r: Radius at given point
        :param W: Cowling Term 1 (Represents lateral fluid flow in the star)
        :param U: Cowling Term 2 (Represents lateral fluid flow in the star)
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param lamda: Metric value for a given mass/radius.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param omega: Oscillation mode as a factor of 2pi
        :param v: Temporal component of metric
        :param c_ad2_inv: Inverse adiabatic sound speed
        :return: Rate of change of Cowling Term 1 wrt rate of change of radius
        """
        c = self.const.c
        term1 = -1 * (l + 1) * (W - l * np.exp(v + lamda / 2) * U)
        frac = -1 * ((omega * r) ** 2) * (np.exp(lamda / 2) * c_ad2_inv)
        term2 = frac * (U - np.exp(lamda / 2) * Q * W * (c ** 2) / ((omega * r) ** 2))
        return term1 + term2

    @staticmethod
    def _dUdlnr(W, U, lamda, l, v):
        """
        First order differential equation for U term in coupled equation (Cowling Approximation 2)
        :param W: Cowling Term 1 (Represents lateral fluid flow in the star)
        :param U: Cowling Term 2 (Represents lateral fluid flow in the star)
        :param lamda: Metric value for a given mass/radius.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param v: Temporal component of metric
        :return: Rate of change of Cowling Term 2 wrt rate of change of radius
        """
        return np.exp(lamda / 2 - v) * (W - l * (np.exp(v - lamda / 2)) * U)

    def _coupledTOV(self, r, VEC, init_params):
        """
        Coupled TOV equation solver in the Cowling approximation.
        :param r: Radius at given point
        :param VEC: Vector containing Pressure, Mass, Nu, Cowling Term 1, Cowling Term 2
        :param init_params: Initializing parameters with given Equation of State (EOS), spherical degree, oscillation
                            mode guess, minimum and maximum pressure.
        :return: Single radial step change in 5 coupled first order differential equation
        """
        # Split vector into seperable components
        P, M, v, W, U = VEC

        # Split Initializing parameters into seperable components
        EOS, l, omega, p_min, p_max = init_params

        # If pressure goes below threshold, then break.
        if P <= p_min:
            return None

        # If pressure goes above threshold, then break.
        if P >= p_max:
            return None

        # If swarzchild criterion exceeds limit, then break.
        if 2 * self._b(r, M) >= 1:
            return None

        # Define argument terms in the coupled ODEs.
        lamda = np.log(1 / (1 - 2 * self._b(r, M)))
        Q = self._Q(r, P, M)
        e = EOS(P)
        c_ad2_inv = self._drhodP(e)

        # Compute single radial step in ODE.
        dPdr = self._dPdr(r, P, M, e)
        dMdr = self._dMdr(r, e)
        dvdr = self._dvdr(r, Q, lamda)
        dWdlnr = self._dWdlnr(r, W, U, Q, lamda, l, omega, v, c_ad2_inv)
        dUdlnr = self._dUdlnr(W, U, lamda, l, v)
        dWdr = dWdlnr * 1 / r
        dUdr = dUdlnr * 1 / r
        ret = [dPdr, dMdr, dvdr, dWdr, dUdr]
        return ret

    def initial_conditions(self, k):
        """
        Define initial conditions for the integration.
        :param k: Index of pressure table used as the initial condition for integration.
        :return:
        """
        self.load_dedp()
        G = self.const.G
        c = self.const.c
        self.r_i = 1  # Initial Radius
        self.p0 = self.p_arr[k]  # Initial pressure
        self.e0 = self.EOS(self.p0)  # Initial energy density

        # Define central pressure and energy density.
        self.p_c = self.p0 - 2 * np.pi * (G / (c ** 4)) * self.r_i ** 2 * (self.p0 + self.e0) * \
                   (3 * self.p0 + self.e0) / 3
        self.e_c = self.EOS(self.p_c)

        self.m0 = self.e_c / (c ** 2) * 4 / 3 * np.pi * self.r_i ** 3  # Central mass density
        self.omega = 2 * (2 * np.pi)  # Initial fmode frquency times 2pi
        self.l = 2  # Spherical oscillation modes
        self.v0 = -1  # Initial metric condition
        self.W0 = 1  # Initial Cowling Term 1
        self.U0 = self.W0 / (self.l * np.exp(self.v0))  # cowling Term 2
        self.init_VEC = [self.p_c, self.m0, self.v0, self.W0, self.U0]  # Initial vector to be passed into TOV equation
        self.p_max = max(self.p_arr)  # Set maximum pressure to be largest value in table
        self.p_min = max(c ** 2, min(self.p_arr))  # Set minimum pressure to be either c^2 or minimum value in table.
        return self.const.km2cm, self.r_i, self.p0, self.e0, self.p_c, self.e_c, self.m0, self.omega, self.l, \
               self.v0, self.W0, self.U0, self.init_VEC, self.p_min, self.p_max

    def tov(self, progress=False):
        """
        Tolman–Oppenheimer–Volkoff (TOV) equation integration wrapper for _coupledTOV()
        :param progress: progress bar
        :return: Integration values (Pressure, Mass, Radius, Temporal Metric, Cowling Term 1, Cowling Term 2)
        """
        init_params = [self.EOS, self.l, self.omega, self.p_min, self.p_max]  # Initial params, per initial conditions.

        # Define ode solver (Older system allows early cutoff)
        r = ode(lambda _r, VEC: self._coupledTOV(_r, VEC, init_params)).set_integrator('VODE')
        r.set_initial_value(self.init_VEC, self.r_i)  # Define ode initial values.

        # Define initial iteration conditions and maximum step radius.
        results = []
        r_list = []
        i = 0
        r_max = 20 * self.const.km2cm
        max_iter = self.n_iter_max
        dr = r_max / max_iter

        # Progress bar
        if progress:
            self.pbar = tqdm(total=max_iter)

        # Compute integration, break if integration is unsuccessful or if pressure is lower than minimum pressure.
        while r.successful() and (r.y[0] >= self.p_min):
            i += 1
            integral = r.integrate(r.t + dr)  # Compute integration step
            if progress:
                self.pbar.update(1)

            # Break if iterations exceeded, pressure is lower than minimum pressure or if integration is unsuccessful.
            if i > max_iter:
                print("[STATUS] max_iter reached")
                break
            if r.y[0] < self.p_min:
                break
            if not r.successful():
                break

            # If all condition passed, update integration values and repeat step.
            results.append(integral)
            r_list.append(r.t + dr)
        if progress:
            self.pbar.close()

        # Update results
        results = np.array(results, dtype=float)
        p, m, v, w, u = results.T
        r = np.array(r_list)
        self.p, self.m, self.v, self.w, self.u, self.r_arr = p, m, v, w, u, r
        return p, m, r, v, w, u

    def update_initial_conditions(self):
        """
        Update initial conditions after first integration to match metrics post integration.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.v, self.w, self.u)

        # Computer external and internal metric, the difference is the change in metric for subsequent integrations.
        v_ext = -self._lamda_metric(m_R, r_R)
        v_int = v_R  # At surface
        delta_v = v_int - v_ext
        self.v0 = self.v0 - delta_v  # Update new nu metric initial condition
        self.U0 = self.W0 / (self.l * np.exp(self.v0))  # Update new Cowling Term 2 in terms of nu initial condition.
        self.init_VEC = [self.p_c, self.m0, self.v0, self.W0, self.U0]  # Send updates to vector used by TOV.
        return None

    def _surface_conditions(self, p, m, r_arr, v, w, u):
        """
        Compute surface condition post integration. All values in units of CGS
        :param p: Integrated pressure values
        :param m: Integrated enclosed mass values
        :param r_arr: Integrated radial values
        :param v: Integrated temporal metric values
        :param w: Integrated Cowling term 1 values
        :param u: Integrated Cowling term 2 values
        :return: Indicies and values at radial integral.
        """
        G = self.const.G
        c = self.const.c
        max_idx = np.argmax(m) - 1  # Reduce by 1 for stability issues.
        m_R = m.max()  # Index of maximum mass in integration
        r_R = r_arr[max_idx]  # Radius at maximum mass
        p_R = p[max_idx]  # Pressure at maximum mass
        ec_R = self.EOS(p_R)  # Energy density at maximum mass
        u_R = u[max_idx]  # Cowling term 1 at maximum mass
        v_R = v[max_idx]  # Metric term at maximum metric
        w_R = w[max_idx]  # Cowling term 2 at maximum mass
        schild = (1 - 2 * G * m_R / (c ** 2 * r_R))
        interior = np.exp(v_R)
        return max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, interior

    def _boundary_wu(self, r_R, m_R, omega, w_R, u_R):
        """
        Boundary condition as defined in arXiv 2205.02081.
        :param r_R: Radius at maximum mass
        :param m_R: Mass at maximum mass index
        :param omega: Oscillation frequency *2pi at maximum mass
        :param w_R: Cowling term 1 at maximum mass
        :param u_R: Cowling term 2 at maximum mass.
        :return: Boundary condition.
        """
        G = self.const.G
        c = self.const.c
        frac1 = (omega ** 2 * r_R ** 3) / (G * m_R)
        return frac1 * np.sqrt(1 - (2 * G * m_R) / (r_R * (c ** 2))) - w_R / u_R

    def print_params(self):
        """
        Print integration parameters.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, interior \
            = self._surface_conditions(self.p, self.m, self.r_arr, self.v, self.w, self.u)
        print("==== INTEGRATION STATS ====")
        print(f"Star has mass {m_R / self.const.msun:.3f} Msun and radius {r_R / self.const.km2cm:.3f}km")
        print(f"Interior Surface: {interior:.8f}")
        print(f"Exterior Surface: {schild:.8f}")
        print(f"v0: {self.v0}")
        print(f"Lamda: {self._lamda_metric(m_R, r_R)}")
        print(f"Boundary Term: {self._boundary_wu(r_R, m_R, self.omega, w_R, u_R)}")
        print()
        return None

    def _minimize_boundary(self, omega):
        """
        Define boundary minimization loss function. (l1 norm- Not l2 norm). Working in log space.
        :param omega: Oscillation frequency * 2pi to guess and compute loss.
        :return: Log-L1 loss function.
        """
        self.omega = omega
        p, m, r_arr, v, w, u = self.tov()  # Complete Integration
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self._surface_conditions(p, m, r_arr, v, w, u) # Compute surface conditions
        loss = np.log10(abs(self._boundary_wu(r_R, m_R, omega, w_R, u_R))) # Compute loss function of this integration

        # Add values of loss for specific oscillation mode.
        self.loss_vals.append(loss)
        self.omega_vals.append(omega)
        return loss

    def _save_mass_radius(self):
        """
        Save values of mass and radius. Not saving other values due to future memory issues in parallelization.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self._surface_conditions(self.p, self.m, self.r_arr, self.v, self.w, self.u)
        self.m_R = m_R
        self.r_R = r_R
        return None

    @staticmethod
    def get_omega_bounds(mass_arr, radius_arr):
        """
        Set bounds on oscillation frequency in the newtonian limit. Values based on lower and upper limit of
        arXiv 1501.02970.
        :param mass_arr: Value of enclosed mass at star surface.
        :param radius_arr: Value of radius at star surface.
        :return: Lower and upper bound of what oscillation modes should be.
        """
        lower = 2 * np.pi * (0.60e3 + 23e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        upper = 2 * np.pi * (0.8e3 + 50e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        return lower, upper

    def optimize_fmode(self, newt_approx=True):
        """
        Run fmode optimizer to determine the fundamental oscillation mode. Non-newtonian optimizer does not use
        oscillation bounds but is more prone to failing.
        :param newt_approx: Determine if newtonian bounds should be applied.
        :return: Optimized fundamental modes.
        """
        self._save_mass_radius() # Determine values of mass and radius at surface.

        if newt_approx:
            # Run optimizer between bounds
            omega_min, omega_max = self.get_omega_bounds(self.m_R, self.r_R)

            # Scalar integration method
            res = minimize_scalar(self._minimize_boundary,
                                  bounds=(omega_min, omega_max),
                                  method='bounded',
                                  options={"maxiter": 30})
            omg = res.x

        else:
            # Run general optimizer (more prone to failing)
            omega_guess = (2 * np.pi) * (0.70e3 + 30e-6 * np.sqrt(self.m_R / (self.r_R ** 3))) # Initial random guess.
            init_guess = [omega_guess]

            # Multivariate integration method
            res = minimize(self._minimize_boundary, x0=init_guess, method='Nelder-Mead',
                           options={"disp": True, "maxiter": 15})
            omg = res.x[0]

        f = omg / (2 * np.pi)
        self.f = f
        self.res = res
        self.omega_arr = np.array(self.omega_vals)
        self.loss_arr = np.array(self.loss_vals)
        return None

    def plot_loss(self):
        """
        Plot loss function of optimizer.
        :return: None
        """
        plt.figure()
        plt.scatter(self.omega_arr / (2 * np.pi), self.loss_arr)
        plt.title(f"fmode: {self.f}")
        plt.show()
        return None

    def plot_pmnuWV(self):
        """
        Plot integration values for pressure, mass, tempoeral metric, Cowling term 1, Cowling term 2.
        :return: None
        """
        r_arr = self.r_arr / self.const.km2cm
        p, m, v, u, w, p_c, = self.p, self.m, self.v, self.u, self.w, self.p_c
        plt.figure()
        plt.plot(r_arr, p / p_c)
        plt.xlabel("r")
        plt.ylabel("P/Pc")

        plt.figure()
        plt.plot(r_arr, m / self.const.msun)
        plt.xlabel("r ")
        plt.ylabel("M/Msun")
        plt.show()

        plt.figure()
        plt.plot(r_arr, v)
        plt.xlabel("r ")
        plt.ylabel("v")
        plt.show()

        plt.figure()
        plt.plot(r_arr, w)
        plt.xlabel("r ")
        plt.ylabel("W")
        plt.show()

        plt.figure()
        plt.plot(r_arr, -u * np.exp(v))
        plt.xlabel("r ")
        plt.ylabel("V")
        plt.show()
        return None


if __name__ == "__main__":
    test = CowlingApproximation()
