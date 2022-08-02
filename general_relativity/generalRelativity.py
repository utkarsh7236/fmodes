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

    @staticmethod
    def _dnudr(r, Q, lamda):
        """
        Change in metric element nu as integration continues through the star.
        :param r: Radius at given point
        :param Q: Transition function dependent on the TOV equations to help determine the metric.
        :param lamda: Metric value for a given mass/radius.
        :return: Temporal metric element.
        """
        return (2 / r) * np.exp(lamda) * Q

    def _H0(self):
        pass

    def _V(self):
        pass

    def _dH1dr(self):
        pass

    def _dKdr(self):
        pass

    def _dWdR(self):
        pass

    def _dXdr(self):
        pass

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
