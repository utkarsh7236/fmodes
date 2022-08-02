#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

import general_relativity.initialize as initialize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import complex_ode
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class GeneralRelativity:
    """
    General Relativity Class for fmode calculations. In the future this can be extended to the generalized GR
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
        self.loss_arr, self.nu0, self.nl, self.X0, self.H00, self.H10 = None, None, None, None, None, None
        self.b0, self.K0, self.Q0, self.r, self.nu, self.h1, self.k = None, None, None, None, None, None, None
        self.x, self.z0, self.z, self.dzdr, self.K0_arr = None, None, None, None, None
        self.hw1, self.hw2, self.hw3 = [300, 100, 30]
        self.omega_vals, self.loss_vals, self.K0_vals = [], [], []

        # Maximum number of integrations
        self.n_iter_max = 20000

        # Set constants and data management system
        self.const = initialize.Constants()
        self.data = initialize.DataManagement()

    @staticmethod
    def _get_ep(e, p):
        """
        Returns Equation of State (EOS) interpolated from the tabular values given.
        :param e: Energy density in units of CGS.
        :param p: Pressure in units of CGS.
        :return: Smoothing functional which interpolates pressure, energy density values.
        """
        f_e_smooth = interp1d(p, e, fill_value="extrapolate", kind="cubic")
        return f_e_smooth

    @staticmethod
    def _get_pe(p, e):
        """
        Returns inverted Equation of State (EOS) interpolated from the tabular values given.
        :param e: Energy density in units of CGS.
        :param p: Pressure in units of CGS.
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
        :param p_arr: Pressure array in CGS.
        :param e_arr: Energy density array in CGS.
        :return: de/dP gradient along with the energy density array used.
        """
        return np.gradient(e_arr, p_arr), e_arr

    def _dedP(self, p_arr, e_arr):
        """
        Compiler function for computing energy density to be used in TOV integration. This equation uses helper
        functions to generate a functional form which predicts a single de/dp given the e, p tabular arrays.
        :param p_arr: Pressure array in CGS.
        :param e_arr: Energy density array in CGS.
        :return: Functional de/dP as a function of e which predicts at a single point of e.
        """
        dedp_helper, e_arr = self._dedP_helper(p_arr, e_arr)
        return interp1d(e_arr, dedp_helper, fill_value="extrapolate", kind="cubic")

    def _drhodP(self, e):
        """
        Convert de/dP into readble CGS units by dividing by c^2.
        :param e: Energy density array in CGS.
        :return: drho/dP which has units of inverse sound speed.
        """
        c = self.const.c
        return (c ** -2) * self.dedp(e)

    def _lamda_metric(self, M, R):
        """
        Compute lamda element of generalized metric by matching to swarzchild metric. Here exp(lamda) = schild.
        :param M: Enclosed Mass.
        :param R: Radius of encolosed mass.
        :return: Lamda value given the swarzchild metric.
        """
        G = self.const.G
        c = self.const.c
        return -np.log((1 - 2 * G * M / (c ** 2 * R)))

    def _dMdr(self, r, e):
        """
        First order mass differential equation (TOV Equation 1)
        :param r: Radius at given point.
        :param e: Energy density at given radius/pressure.
        :return: Rate of change of mass/radius.
        """
        c = self.const.c
        return 4 * np.pi * r ** 2 * (e / (c ** 2))

    def _b(self, r, M):
        """
        Functional of swarchild as defined in arXiv 2205.02081, ignored factor of 2 due to their typo.
        :param r: Radius at given point.
        :param M: Enclosed mass.
        :return: Functional value of mass/radius dependent term in swarzchild.
        """
        G = self.const.G
        c = self.const.c
        return (G * M) / ((c ** 2) * r)

    def _Q(self, r, P, M):
        """
        Transition function, dependent on the TOV equations.
        :param r: Radius at given point.
        :param P: Pressure at a given radius.
        :param M: Enclosed mass.
        :return: Functional value of TOV transition function as determined by arXiv 2205.02081.
        """
        G = self.const.G
        c = self.const.c
        frac = (4 * np.pi * G * (r ** 2) * P) / (c ** 4)
        return self._b(r, M) + frac

    def _dPdr(self, r, P, M, e):
        """
        First order pressure differential equation (TOV Equation 2)
        :param r: Radius at given point.
        :param P: Pressure at a given radius.
        :param M: Enclosed mass.
        :param e: Energy density at a given pressure.
        :return: Rate of change of pressure/radius.
        """
        G = self.const.G
        c = self.const.c
        num = (M + 4 * np.pi * (r ** 3) * P / (c ** 2))
        dem = r * (r - 2 * G * M / (c ** 2))
        return -1 * (e + P) * G / (c ** 2) * num / dem

    @staticmethod
    def _dnudr(r, Q, lamda):
        """
        Change in metric element nu as integration continues through the star.
        :param r: Radius at given point.
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param lamda: Metric value for a given mass/radius.
        :return: Temporal metric element.
        """
        return (2 / r) * np.exp(lamda) * Q

    def _H0(self, r, nu, X, nl, Q, omega, lamda, H1, b, K):
        """
        :param r: Radius at given point.
        :param nu: Temporal component of metric.
        :param X: Lagrangian pressure variation.
        :param nl: Factor of order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param omega: Oscillation mode as a factor of 2pi on the INTERIOR of the star.
        :param lamda: Metric value for a given mass/radius.
        :param H1: Fluid perturbations of star.
        :param b: Compactness (appears in the Swarzchild Metric).
        :param K: Fluid perturbations of star.
        :return: Initial fluid perturbation H0 (Diagaonal perturbation).
        """
        G = self.const.G
        c = self.const.c
        omgr2_c2 = ((omega * r) ** 2) / (c ** 2)
        factor = (2 * b + nl + Q) ** (-1)
        term1 = 8 * np.pi * (r ** 2) * np.exp(-nu / 2) * X * (G / (c ** 4))
        term2 = - H1 * (Q * (nl + 1) - omgr2_c2 * np.exp(-nu - lamda))
        term3 = K * (nl - omgr2_c2 * np.exp(-nu) - Q * (np.exp(lamda) * Q - 1))
        return factor * (term1 + term2 + term3)

    def _V(self, r, X, e, p, Q, nu, lamda, W, H0, omega):
        """
        :param r: Radius at given point.
        :param X: Lagrangian pressure variation.
        :param e: Energy density of neutron star at a given radius.
        :param p: Pressure of neutron star at a given radius.
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param nu: Temporal component of metric.
        :param lamda: Metric value for a given mass/radius.
        :param W: Fluid perturbations of star.
        :param H0: Fluid perturbations of star.
        :param omega: Oscillation mode as a factor of 2pi on the INTERIOR of the star.
        :return: Fluid perturbation element (spherical).
        """
        c = self.const.c
        factor = np.exp(nu / 2) * (c ** 2) / (omega ** 2)
        term1 = X / (e + p)
        term2 = -Q / (r ** 2) * W * np.exp((nu + lamda) / 2)
        term3 = -np.exp(nu / 2) * H0 / 2
        return factor * (term1 + term2 + term3)

    def _dH1dr(self, r, l, b, lamda, p, e, H1, H0, K, V):
        """
        :param r: Radius at given point.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param b: Compactness (appears in the Swarzchild Metric).
        :param lamda: Metric value for a given mass/radius.
        :param e: Energy density of neutron star at a given radius.
        :param p: Pressure of neutron star at a given radius.
        :param H1: Fluid perturbations of star.
        :param H0: Fluid perturbations of star.
        :param K: Fluid perturbations of star.
        :param V: Fluid perturbations of star.
        :return: Fluid perturbations of star (off-diagonal).
        """
        G = self.const.G
        c = self.const.c
        G_c4 = G / (c ** 4)
        term1 = -H1 * (l + 1 + 2 * b * np.exp(lamda) + G_c4 * 4 * np.pi * (r ** 2) * np.exp(lamda) * (p - e))
        term2 = np.exp(lamda) * (H0 + K - G_c4 * 16 * np.pi * (e + p) * V)
        return (1 / r) * (term1 + term2)

    def _dKdr(self, r, H0, nl, H1, lamda, Q, l, K, e, p, W):
        """
        :param r: Radius at given point.
        :param H0: Fluid perturbations of star.
        :param nl: Factor of order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param H1: Fluid perturbations of star.
        :param lamda: Metric value for a given mass/radius.
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param K: Fluid perturbations of star.
        :param e: Energy density of neutron star at a given radius.
        :param p: Pressure of neutron star at a given radius.
        :param W: Fluid perturbations of star.
        :return: Fluid perturbations of star (spherical diagonal).
        """
        G = self.const.G
        c = self.const.c
        term1 = H0
        term2 = (nl + 1) * H1
        term3 = (np.exp(lamda) * Q - l - 1) * K
        term4 = -8 * np.pi * (e + p) * np.exp(lamda / 2) * W * G / (c ** 4)
        return (1 / r) * (term1 + term2 + term3 + term4)

    def _dWdR(self, r, W, l, lamda, V, e, p, X, cad2_inv, H0, K, nu):
        """
        :param r: Radius at given point.
        :param W: Fluid perturbations of star.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param lamda: Metric value for a given mass/radius.
        :param V: Fluid perturbations of star.
        :param e: Energy density of neutron star at a given radius.
        :param p: Pressure of neutron star at a given radius.
        :param X: Lagrangian pressure variation.
        :param cad2_inv: Speed of sound of the fluid.
        :param H0: Fluid perturbations of star.
        :param K: Fluid perturbations of star.
        :param nu: Temporal component of metric.
        :return: Fluid perturbations of star (radial perturbation).
        """
        G = self.const.G
        c = self.const.c
        term1 = -(l + 1) * (W + l * np.exp(lamda / 2) * V)
        term2_fac = (r ** 2 * np.exp(lamda / 2))
        term2_1 = (np.exp(-nu / 2) * X * (c ** 2) * cad2_inv) / (e + p)
        term2_2 = H0 / 2
        term2_3 = K
        term2 = term2_fac * (term2_1 + term2_2 + term2_3)
        return (1 / r) * (term1 + term2)

    def _dXdr(self, r, l, X, e, p, nu, lamda, Q, H0, omega, nl, H1, K, V, W, derivative_term):
        """
        :param r: Radius at given point.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param X: Lagrangian pressure variation.
        :param e: Energy density of neutron star at a given radius.
        :param p: Pressure of neutron star at a given radius.
        :param nu: Temporal component of metric.
        :param lamda: Metric value for a given mass/radius.
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param H0:mFluid perturbations of star.
        :param omega: Oscillation mode as a factor of 2pi on the INTERIOR of the star.
        :param nl: Factor of order of term in spherical harmonics, l = 2 corresponds to the primary oscillation modes.
        :param H1: Fluid perturbations of star.
        :param K: Fluid perturbations of star.
        :param V: Fluid perturbations of star.
        :param W: Fluid perturbations of star.
        :param derivative_term: Term invovlving derivative of arXiv 2205.02081 EQ14.
        :return: Lagrangian pressure variation.
        """
        G = self.const.G
        c = self.const.c
        omgr2_c2 = ((omega * r) ** 2) / (c ** 2)
        G_c4 = G / (c ** 4)
        term1 = -l * X
        term2_fac = (e + p) * np.exp(nu / 2) / 2
        term2_1 = (1 - np.exp(lamda) * Q) * H0
        term2_2 = H1 * (omgr2_c2 * np.exp(-nu) + nl + 1)
        term2_3 = K * (3 * np.exp(lamda) * Q - 1)
        term2_4 = -4 * (nl + 1) * np.exp(lamda) * Q * V / (r ** 2)
        term2_5_fac = -2 * W
        term2_5_1 = np.exp(lamda / 2 - nu) * (omega ** 2) / (c ** 2)
        term2_5_2 = G_c4 * 4 * np.pi * (e + p) * np.exp(lamda / 2)
        term2_5_3 = -(r ** 2) * derivative_term
        term2_5 = term2_5_fac * (term2_5_1 + term2_5_2 + term2_5_3)
        term2 = term2_fac * (term2_1 + term2_2 + term2_3 + term2_4 + term2_5)
        return (1 / r) * (term1 + term2)

    def derivative_term(self, r, dMdr, dPdr, m, P):
        """ Manually taken the implicit derivative taken from arXiv 2205.02081 EQ14.
        :param r: Radius at given point.
        :param dMdr: Change is mass at given radius.
        :param dPdr: Change in pressure at given radius.
        :param m: Mass at given radius.
        :param P: Pressure at given radius.
        :return:
        """
        G = self.const.G
        c = self.const.c
        pi = np.pi
        exp = np.exp
        sqrt = np.sqrt
        return G * (G * (dMdr * r - m) * (4 * pi * P * r ** 3 + c ** 2 * m) - 6 * (-2 * G * m + c ** 2 * r) * (
                4 * pi * P * r ** 3 + c ** 2 * m) + 2 * (-2 * G * m + c ** 2 * r) * (
                            c ** 2 * dMdr * r - c ** 2 * m + 4 * pi * r ** 3 * (2 * P + dPdr * r))) / (
                       2 * c ** 6 * r ** 6 * ((-2 * G * m + c ** 2 * r) / (c ** 2 * r)) ** (5 / 4))

    def _coupledTOV(self, r, VEC, init_params):
        P, M, nu, H1, K, W, X = VEC
        EOS, l, omega, p_min, p_max, nl = init_params

        if P <= p_min:
            return None
        if P >= p_max:
            return None
        if 2 * self._b(r, M) >= 1:
            return None

        b = self._b(r, M)
        lamda = np.log(1 / (1 - 2 * b))
        Q = self._Q(r, P, M)
        e = EOS(np.real(P))
        cad2_inv = self._drhodP(e)

        dPdr = self._dPdr(r, P, M, e)
        dMdr = self._dMdr(r, e)

        derv_term = self.derivative_term(r, dMdr, dPdr, M, P)

        dnudr = self._dnudr(r, Q, lamda)
        H0 = self._H0(r, nu, X, nl, Q, omega, lamda, H1, b, K)
        V = self._V(r, X, e, P, Q, nu, lamda, W, H0, omega)

        # arXiv 2205.02081
        dH1dr = self._dH1dr(r, l, b, lamda, P, e, H1, H0, K, V)
        dKdr = self._dKdr(r, H0, nl, H1, lamda, Q, l, K, e, P, W)
        dWdr = self._dWdR(r, W, l, lamda, V, e, P, X, cad2_inv, H0, K, nu)
        dXdr = self._dXdr(r, l, X, e, P, nu, lamda, Q, H0, omega, nl, H1, K, V, W, derv_term)

        ret = [dPdr, dMdr, dnudr, dH1dr, dKdr, dWdr, dXdr]
        return ret

    def initial_conditions_helper(self, K0, e_c, p_c, nu0, omega, W0, nl, l, p0, e0):
        G = self.const.G
        c = self.const.c
        e_c = e0
        p_c = p0
        X0_factor = (e_c + p_c) * np.exp(nu0 / 2)
        X0_term1 = 4 * np.pi / 3 * (e_c + 3 * p_c) * W0 * G / (c ** 4)
        X0_term2 = -(omega ** 2) / l * np.exp(-nu0) * W0 / (c ** 2)
        X0_term3 = K0 / 2
        X0 = X0_factor * (X0_term1 + X0_term2 + X0_term3)
        H00 = K0
        H10 = (l * K0 + 8 * np.pi * (G / (c ** 4)) * (e_c + p_c) * W0) / (nl + 1)
        return X0, H00, H10

    def initial_conditions(self, k):
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
        self.omega = 2e3 * (2 * np.pi) + 0.1j # Initial fmode frquency times 2pi

        self.l = 2  # Spherical oscillation modes
        self.nu0 = -1  # Initial metric condition
        self.W0 = 1  # Initial Cowling Term 1
        self.nl = (self.l - 1) * (self.l + 2) / 2
        self.Q0 = self._Q(self.r_i, self.p_c, self.m0)
        self.b0 = self._b(self.r_i, self.m0)
        self.W0 = 1
        self.K0 = -(self.e_c + self.p_c) * (G / (c ** 4))

        self.X0, self.H00, self.H10 = self.initial_conditions_helper(self.K0, self.e_c, self.p_c, self.nu0, self.omega,
                                                                     self.W0, self.nl, self.l, self.p0, self.e0)
        self.init_VEC = [self.p_c, self.m0, self.nu0, self.H10, self.K0, self.W0, self.X0]
        self.p_max = max(self.p_arr)  # Set maximum pressure to be largest value in table
        self.p_min = max(c ** 2, min(self.p_arr))  # Set minimum pressure to be either c^2 or minimum value in table.
        return self.p_c, self.e_c, self.m0, self.omega, self.l, self.nl, self.nu0, self.Q0, self.b0, self.W0, self.K0, \
               self.X0, self.H00, self.H10, self.init_VEC, self.p_max, self.p_min, self.r_i, self.p0, self.e0

    def tov(self, progress=False):
        init_params = [self.EOS, self.l, self.omega, self.p_min, self.p_max, self.nl]
        r = complex_ode(lambda _r, VEC: self._coupledTOV(_r, VEC, init_params)).set_integrator('VODE')
        r.set_initial_value(self.init_VEC, self.r_i)
        results = [self.init_VEC]
        r_list = [self.r_i]
        i = 0
        r_max = 20 * self.const.km2cm
        max_iter = self.n_iter_max
        dr = r_max / max_iter
        m_max = 0
        if progress:
            self.pbar = tqdm(total=max_iter)
        while r.successful() and (np.real(r.y[0]) >= self.p_min):
            i += 1
            integral = r.integrate(r.t + dr)
            if progress:
                self.pbar.update(1)
            if i > max_iter:
                print("[STATUS] max_iter reached")
                break
            if np.real(r.y[0]) < self.p_min:
                break
            if not r.successful():
                break
            results.append(integral)
            r_list.append(r.t + dr)

            if r.y[1] < m_max:
                break

            m_max = max(r.y[1], m_max)

        if progress:
            self.pbar.close()

        results = np.array(results, dtype=complex)
        p, m, nu, h1, k, w, x = results.T
        r = np.array(r_list, dtype=float)
        self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x = p, m, r, nu, h1, k, w, x
        return p, m, r, nu, h1, k, w, x

    def update_initial_conditions(self):
        """
        Update initial conditions after first integration to match metrics post integration.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, interior \
            = self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)

        # Computer external and internal metric, the difference is the change in metric for subsequent integrations.
        nu_ext = -self._lamda_metric(m_R, r_R)
        nu_int = nu_R  # At surface
        delta_nu = nu_int - nu_ext
        self.nu0 = self.nu0 - delta_nu
        self.X0, self.H00, self.H10 = self.initial_conditions_helper(self.K0, self.e_c, self.p_c, self.nu0, self.omega,
                                                                     self.W0, self.nl, self.l, self.p0, self.e0)
        self.init_VEC = [self.p_c, self.m0, self.nu0, self.H10, self.K0, self.W0, self.X0]
        return None

    def _surface_conditions(self, p, m, r_arr, nu, h1, k, w, x):
        G = self.const.G
        c = self.const.c
        max_idx = np.argmax(m) - 1
        m_R = m.max()  # In units of msun
        r_R = r_arr[max_idx]  # In units of km
        p_R = p[max_idx]  # cgs
        ec_R = self.EOS(np.real(p_R))  # cgs
        nu_R = nu[max_idx]
        h1_R = h1[max_idx]
        k_R = k[max_idx]
        w_R = w[max_idx]
        x_R = x[max_idx]
        schild = (1 - 2 * G * m_R / (c ** 2 * r_R))
        interior = np.exp(nu_R)
        return max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, interior

    def print_params(self):
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)
        print(f"Star has mass {m_R / self.const.msun:.3f} Msun and radius {r_R / self.const.km2cm:.3f}km")
        print(f"Interior Surface: {interior:.8f}")
        print(f"Exterior Surface: {schild:.8f}")
        print(f"v0: {self.nu0}")
        print(f"Lamda: {self._lamda_metric(m_R, r_R)}")
        print(f"Boundary Term: {x_R}")
        return None

    def optimize_x_R(self, K0):
        # Update Initial Conditions in terms of K0
        self.X0, self.H00, self.H10 = self.initial_conditions_helper(K0, self.e_c, self.p_c, self.nu0, self.omega,
                                                                     self.W0,
                                                                     self.nl, self.l, self.p0, self.e0)
        self.init_VEC = np.array([self.p_c, self.m0, self.nu0, self.H10, K0, self.W0, self.X0],
                                 dtype=complex).flatten()
        p, m, r_arr, nu, h1, k, w, x = self.tov()
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, \
        interior = self._surface_conditions(p, m, r_arr, nu, h1, k, w, x)
        loss = np.log10(abs(x_R))  # Absolute loss of complex B.C.
        self.loss_vals.append(loss)
        self.K0_vals.append(K0)
        return loss

    def minimize_K0(self):
        self.n_iter_max = 10000
        G = self.const.G
        c = self.const.c
        K0_guess = self.K0
        init_guess = [K0_guess]

        res = minimize(self.optimize_x_R, x0 = init_guess, method='Nelder-Mead',
                       options = {"disp": True, "xatol":1e-3, "fatol":1e-3})

        K0 = res.x[0]
        X0_factor = (self.e_c + self.p_c) * np.exp(self.nu0 / 2)
        X0_term1 = 4 * np.pi / 3 * (self.e_c + 3 * self.p_c) * self.W0 * G / (c ** 4)
        X0_term2 = -(self.omega ** 2) / self.l * np.exp(-self.nu0) * self.W0 / (c ** 2)
        X0_term3 = K0 / 2
        self.K0 = K0
        self.X0 = X0_factor * (X0_term1 + X0_term2 + X0_term3)
        self.H00 = K0
        self.H10 = (self.l * K0 + 8 * np.pi * (G / (c ** 4)) * (self.e_c + self.p_c) * self.W0) / (self.nl + 1)
        self.n_iter_max = 20000
        self.K0_arr = np.array(self.K0_vals)
        self.loss_arr = np.array(self.loss_vals)
        return None

    def plot_loss(self):
        plt.figure()
        plt.scatter(self.K0_arr, self.loss_arr)
        plt.title(f"K0 Minimized: {self.K0}")
        plt.show()
        return None

    def _save_mass_radius(self):
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)
        self.m_R = m_R
        self.r_R = r_R
        return None

    def plot_fluid_perturbations(self):
        r_arr = self.r_arr
        km2cm = self.const.km2cm
        lims = 1

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.p[lims:]) / self.p_c)
        plt.xlabel("r")
        plt.ylabel("P/Pc")

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.m[lims:]) / self.const.msun)
        plt.xlabel("r ")
        plt.ylabel("M/Msun")
        plt.show()

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.nu[lims:]))
        plt.xlabel("r ")
        plt.ylabel("v")
        plt.show()

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.h1[lims:]))
        plt.xlabel("r ")
        plt.ylabel("H1")
        plt.show()

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, -np.real(self.k[lims:]))
        plt.xlabel("r ")
        plt.ylabel("-K")
        plt.show()

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.w[lims:]))
        plt.xlabel("r ")
        plt.ylabel("W")
        plt.show()

        plt.figure()
        plt.plot(r_arr[lims:] / km2cm, np.real(self.x[lims:]))
        plt.xlabel("r ")
        plt.ylabel("X")
        plt.show()
        return None

    # def zerilli_alt(self, r_star, u, omega):
    #     z, dZdr_star = u
    #     d2Zdr2_star = z * (self.V_z_alt(r_star, self.m_R, self.nl) - omega * np.conj(omega) / (self.const.c ** 2))
    #     ret = [dZdr_star, d2Zdr2_star]
    #     return ret
    #
    # def alphas(self, omega, m_R):
    #     n = self.nl
    #     G = self.const.G
    #     c = self.const.c
    #     alpha0 = 1 + 1j
    #     alpha1 = -1j * (n + 1) * alpha0 * c / omega
    #     alpha2 = alpha0 * (c ** 2) * (-n * (n + 1) + 1j * m_R * omega * (G / (c ** 3)) * (3 / 2 + 3 / n)) / (
    #                 2 * omega ** 2)
    #     return alpha0, alpha1, alpha2
    #
    # def V_z_alt(self, r, m_R, nl):
    #     b = self._b(r, m_R)
    #     n = self, nl
    #     fac = (1 - 2 * b)
    #     num = 2 * (n ** 2) * (n + 1) + 6 * (n ** 2) * b + 18 * n * (b ** 2) + 18 * (b ** 3)
    #     dem = (r ** 2) * (n + 3 * b) ** 2
    #     return fac * num / dem
    #
    # def r_star_func(self, r, m_R):
    #     G = self.const.G
    #     c = self.const.c
    #     return r + 2 * (G / (c ** 2)) * m_R * np.log(abs((r * (c ** 2)) / (2 * G * m_R) - 1))
    #
    # def V_z(self, r, m_R):
    #     nl = self.nl
    #     G = self.const.G
    #     c = self.const.c
    #     G_c2 = G / (c ** 2)
    #     num = (1 - 2 * G_c2 * m_R / r)
    #     dem = (r ** 3) * ((nl * r + 3 * G_c2 * m_R) ** 2)
    #     fac1 = 2 * nl ** 2 * (nl + 1) * (r ** 3)
    #     fac2 = 6 * (G_c2 ** 1) * (nl ** 2) * m_R * (r ** 2)
    #     fac3 = 18 * (G_c2 ** 2) * nl * (m_R ** 2) * r
    #     fac4 = 18 * (G_c2 ** 3) * (m_R ** 3)
    #     fac = fac1 + fac2 + fac3 + fac4
    #     ret = fac * num / dem
    #     return ret
    #
    # def zerilli(self, r_star, u, omega):
    #     omega2 = pow(omega, 2)
    #     z, dZdr_star = u
    #     d2Zdr2_star = z * (self.V_z(r_star, self.m_R) - omega2 / (self.const.c ** 2))
    #     ret = [dZdr_star, d2Zdr2_star]  # dZ/dr*, d2Z/dr*2
    #     return ret
    #
    # def zrly(self, omega, r_star_vals, progress=True):
    #     r = complex_ode(lambda r, VEC: self.zerilli(r, VEC, omega)).set_integrator('LSODA', atol=1.49012e-8,
    #                                                                                rtol=1.49012e-8)
    #     r.set_initial_value(self.z0, self.r_star_func(self.r_R, np.real(self.m_R)))
    #     results = [self.z0]
    #     r_list = [self.r_star_func(self.r_R, self.m_R)]
    #     i = 0
    #     if progress:
    #         self.pbar = tqdm(total=len(r_star_vals))
    #     while r.successful():
    #         i += 1
    #         if i >= len(r_star_vals):
    #             break
    #         if progress:
    #             self.pbar.update(1)
    #
    #         integral = r.integrate(r_star_vals[i])
    #         if not r.successful():
    #             break
    #         results.append(integral)
    #         r_list.append(r_star_vals[i])
    #     results = np.array(results, dtype=complex)
    #     z, dzdr = results.T
    #     r = np.array(r_list)
    #     self.z, self.dzdr, self.r = z, dzdr, r
    #     return z, dzdr, r
    #
    # def optimize_fmode(self):
    #     pass
    #
    # def plot_loss_fmode(self):
    #     pass


if __name__ == "__main__":
    test = GeneralRelativity()
