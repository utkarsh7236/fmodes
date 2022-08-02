#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

import general_relativity.initialize as initialize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import ode
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar
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

    def derivative_term(self):
        pass

    def _coupledTOV(self):
        pass

    def initial_conditions(self):
        pass

    def tov(self):
        pass

    def update_initial_conditions(self):
        pass

    def _surface_conditions(self):
        pass

    def print_params(self):
        pass

    def minimize_K0(self):
        pass

    def _save_mass_radius(self):
        pass

    def optimize_fmode(self):
        pass

    def plot_fluid_perturbations(self):
        pass


if __name__ == "__main__":
    test = GeneralRelativity()
