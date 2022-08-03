#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

import general_relativity.initialize as initialize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import complex_ode
from tqdm import tqdm
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import time


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
        self.x, self.z0, self.z, self.dzdr, self.K0_arr, self.fguess = None, None, None, None, None, None
        self.imag, self.main_index_k, self.fmode_arr, self.abs_loss_arr, self.loss_arr2 = None, None, None, None, None
        self.f_arr, self.popt, self.fmode = None, None, None
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
        """
        Helper function for full TOV integration. Evaluates and updates initial step for each coupled differential
        equation.
        :param r: Radius at a given point during the integration. (Begins near 0)
        :param VEC: Vector of variables integrated, updated variables are dependent on previous integrations.
        :param init_params: Initial argument passed into helper function which define certain constrains on integration.
        :return: Updated step changes for a given radius at a given step side dr.
        """

        # Extract pressure/fluid perturbations from vector.
        P, M, nu, H1, K, W, X = VEC

        # Extract arguments used as settings in the integration.
        EOS, l, omega, p_min, p_max, nl = init_params

        # Define exit conditions for integration; minimum/maximum pressure and undefined compactness.
        if P <= p_min:
            return None
        if P >= p_max:
            return None
        if 2 * self._b(r, M) >= 1:
            return None

        # Extract transition terms using helper functions.
        b = self._b(r, M)
        lamda = np.log(1 / (1 - 2 * b))
        Q = self._Q(r, P, M)
        e = EOS(np.real(P))

        # Determine energy density and speed of sound.
        cad2_inv = self._drhodP(e)

        # Update change in pressure and mass (Original TOV).
        dPdr = self._dPdr(r, P, M, e)
        dMdr = self._dMdr(r, e)
        dnudr = self._dnudr(r, Q, lamda)

        # Extract derivative term as per arXiv 2204.03037.
        derv_term = self.derivative_term(r, dMdr, dPdr, M, P)

        # Update static fluid perturbations
        H0 = self._H0(r, nu, X, nl, Q, omega, lamda, H1, b, K)
        V = self._V(r, X, e, P, Q, nu, lamda, W, H0, omega)

        # Update dynamic fluid perturbations as per arXiv 2205.02081
        dH1dr = self._dH1dr(r, l, b, lamda, P, e, H1, H0, K, V)
        dKdr = self._dKdr(r, H0, nl, H1, lamda, Q, l, K, e, P, W)
        dWdr = self._dWdR(r, W, l, lamda, V, e, P, X, cad2_inv, H0, K, nu)

        # Update Lagrangian pressure perturbation.
        dXdr = self._dXdr(r, l, X, e, P, nu, lamda, Q, H0, omega, nl, H1, K, V, W, derv_term)

        # Return all changes to VEC.
        ret = [dPdr, dMdr, dnudr, dH1dr, dKdr, dWdr, dXdr]
        return ret

    def initial_conditions_helper(self, K0, e_c, p_c, nu0, omega, W0, nl, l, p0, e0):
        """
        Helper function used to define initial conditions of fluid and Lagrangian perturbations.
        :param K0: Initial value for fluid perturbation (Guessed as per arXiv 2204.03037).
        :param e_c: Central energy density (Different from e0)
        :param p_c: Central pressure (Different from p0)
        :param nu0: Temporal metric component in the background solution.
        :param omega: Interior oscillation mode frequency multipled by 2 pi.
        :param W0: Initial value for fluid perturbation (Trivially set to 1)
        :param nl: Factor of order of term in spherical harmonics, l = 2 corresponds to the primary oscillation mode.
        :param l: Order of term in spherical harmonics, l = 2 corresponds to the primary oscillation mod
        :param p0: Pressure at r = 0 (Different from central pressure)
        :param e0: Energy density at r = 0 (Different from central energy density)
        :return: Initial condition for lagrangian pressure variation X0, and fluid perturbations H00, H10.
        """
        # Define constants
        G = self.const.G
        c = self.const.c

        # Set central pressure equal to pressure at r = 0.
        e_c = e0
        p_c = p0

        # Compute lagrangian pressure variations
        X0_factor = (e_c + p_c) * np.exp(nu0 / 2)
        X0_term1 = 4 * np.pi / 3 * (e_c + 3 * p_c) * W0 * G / (c ** 4)
        X0_term2 = -(omega ** 2) / l * np.exp(-nu0) * W0 / (c ** 2)
        X0_term3 = K0 / 2
        X0 = X0_factor * (X0_term1 + X0_term2 + X0_term3)

        # Update fluid perturbation.
        H00 = K0
        H10 = (l * K0 + 8 * np.pi * (G / (c ** 4)) * (e_c + p_c) * W0) / (nl + 1)
        return X0, H00, H10

    def initial_conditions(self, k):
        """
        Set initial conditions to be used in the TOV integration.
        :param k: Index of pressure table in which pressure values are to be used. (Index for initial P, e)
        :return: All initial conditions for the integration.
        """
        # Load pressure tables and constants
        self.load_dedp()
        G = self.const.G
        c = self.const.c

        self.r_i = 1  # Initial Radius
        self.p0 = self.p_arr[k]  # Initial pressure
        self.e0 = self.EOS(self.p0)  # Initial energy density

        self.main_index_k = k  # Define main index to be used in Zerilli equation.

        # Define central pressure and energy density.
        self.p_c = self.p0 - 2 * np.pi * (G / (c ** 4)) * self.r_i ** 2 * (self.p0 + self.e0) * \
                   (3 * self.p0 + self.e0) / 3
        self.e_c = self.EOS(self.p_c)

        # Define initial mass
        self.m0 = self.e_c / (c ** 2) * 4 / 3 * np.pi * self.r_i ** 3  # Central mass density

        # Initial value used for omega (Should be arbitrary but reasonable).
        self.imag = 0.1j
        self.omega = 2e3 * (2 * np.pi) + self.imag  # Initial fmode frquency times 2pi

        self.l = 2  # Spherical oscillation modes
        self.nu0 = -1  # Initial metric condition
        self.W0 = 1  # Initial Cowling Term 1

        # Transition functions and factors
        self.nl = (self.l - 1) * (self.l + 2) / 2
        self.Q0 = self._Q(self.r_i, self.p_c, self.m0)
        self.b0 = self._b(self.r_i, self.m0)

        # Initial fluid perturbations
        self.W0 = 1
        self.K0 = -(self.e_c + self.p_c) * (G / (c ** 4))

        self.X0, self.H00, self.H10 = self.initial_conditions_helper(self.K0, self.e_c, self.p_c, self.nu0, self.omega,
                                                                     self.W0, self.nl, self.l, self.p0, self.e0)

        # Assign vector with useful terms in "VEC" section of integration.
        self.init_VEC = [self.p_c, self.m0, self.nu0, self.H10, self.K0, self.W0, self.X0]
        self.p_max = max(self.p_arr)  # Set maximum pressure to be largest value in table
        self.p_min = max(c ** 2, min(self.p_arr))  # Set minimum pressure to be either c^2 or minimum value in table.
        return self.p_c, self.e_c, self.m0, self.omega, self.l, self.nl, self.nu0, self.Q0, self.b0, self.W0, self.K0, \
               self.X0, self.H00, self.H10, self.init_VEC, self.p_max, self.p_min, self.r_i, self.p0, self.e0

    def tov(self, progress=False):
        """
        Main TOV integraiton for background solution and fluid perturbations to the fully GR case.
        :param progress: Should the program display integration progress.
        :return: Values of VEC along each point in the star (radially).
        """

        # Set initial arguments used in the integration
        init_params = [self.EOS, self.l, self.omega, self.p_min, self.p_max, self.nl]

        # Define complex ODE solver using the VODE method, pass initial argument.
        r = complex_ode(lambda _r, VEC: self._coupledTOV(_r, VEC, init_params)).set_integrator('VODE')

        # Set initial radius used to integrate.
        r.set_initial_value(self.init_VEC, self.r_i)

        # Include first step of integration
        results = [self.init_VEC]
        r_list = [self.r_i]

        # Set index and step size.
        i = 0
        r_max = 20 * self.const.km2cm
        max_iter = self.n_iter_max
        dr = r_max / max_iter

        # Initial early stopping according to mass.
        m_max = 0

        # Initialize progress bar and integration
        if progress:
            self.pbar = tqdm(total=max_iter)

        # Keep looping while integration is successful and pressure has not reached "0". (Orders of magnitude lower)
        while r.successful() and (np.real(r.y[0]) >= self.p_min):
            i += 1
            integral = r.integrate(r.t + dr)  # Update integral
            if progress:
                self.pbar.update(1)

            # Define break conditions. Exit loop if integration was not successful or if pressure exceeds defined
            # limits. Only consider real values for pressure, only fluid perturbations may be imaginary.
            if i > max_iter:
                print("[STATUS] max_iter reached")
                break
            if np.real(r.y[0]) < self.p_min:
                break
            if not r.successful():
                break

            # If integration succeeds, then append values of integration to saved list.
            results.append(integral)
            r_list.append(r.t + dr)

            # Mass decrease integration exit conditions step
            if r.y[1] < m_max:
                break

            # Mass decrease integration update step
            m_max = max(r.y[1], m_max)

        if progress:
            self.pbar.close()

        # Complete integration and save data.
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
        """
        Define surface conditions at the end of TOV integration.
        :param p: Pressure array for all radii
        :param m: Mass array for all radii
        :param r_arr: Radias array for which all other arrays are defined in terms of.
        :param nu: Temporal metric for all radii.
        :param h1: Fluid perturbation for all radii.
        :param k: Fluid perturbation for all radii.
        :param w: Fluid perturbation for all radii.
        :param x: Lagranging pressure preturbation for all radii.
        :return: Values of VEC at the maximum radius including the interior and exterior Swarzchild metric.
        """
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
        """ Print parameters extracted from the surface conditions in a readable format.
        Includes mass, radius and other common components.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)
        print(f"Star has mass {m_R / self.const.msun:.3f} Msun and radius {r_R / self.const.km2cm:.3f}km")
        print(f"Interior Surface: {interior:.8f}")
        print(f"Exterior Surface: {schild:.8f}")
        print(f"nu0: {self.nu0}")
        print(f"Lamda: {self._lamda_metric(m_R, r_R)}")
        print(f"Boundary Term: {x_R}")
        return None

    def optimize_x_R(self, K0):
        """ Helper function to be used in optimizing the fluid perturbation.
        :param K0: Iniital fluid perturbation (Arbitrary multiple of e_c, p_c).
        :return: Loss for a given initial fluid perturbation.
        """
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
        """
        Optimize initial fluid perturbation in terms of a given boundary condition.
        Lagrangian pressure variations should tend to zero at the surface. Refer to arXiv 2205.02081.
        :return:
        """
        self.n_iter_max = 5000 # Reduced for faster integration times.
        G = self.const.G
        c = self.const.c

        # Define an initial guess as per arXiv 2205.02081.
        K0_guess = self.K0
        init_guess = [K0_guess]

        # Apply minimization function to optimize boundary condition.
        res = minimize(self.optimize_x_R, x0=init_guess, method='Nelder-Mead',
                       options={"disp": False, "xatol": 1e-3, "fatol": 1e-3})

        # Update initial conditions for all fluid perturbations.
        K0 = res.x[0]
        X0_factor = (self.e_c + self.p_c) * np.exp(self.nu0 / 2)
        X0_term1 = 4 * np.pi / 3 * (self.e_c + 3 * self.p_c) * self.W0 * G / (c ** 4)
        X0_term2 = -(self.omega ** 2) / self.l * np.exp(-self.nu0) * self.W0 / (c ** 2)
        X0_term3 = K0 / 2
        self.K0 = K0
        self.X0 = X0_factor * (X0_term1 + X0_term2 + X0_term3)
        self.H00 = K0
        self.H10 = (self.l * K0 + 8 * np.pi * (G / (c ** 4)) * (self.e_c + self.p_c) * self.W0) / (self.nl + 1)

        # Reset integration for future use.
        self.n_iter_max = 20000
        self.K0_arr = np.array(self.K0_vals)

        # Keep track of loss through optimization.
        self.loss_arr = np.array(self.loss_vals)
        return None

    def plot_loss(self):
        """
        Graphic representation of loss as a function of optimization steps in K0.
        :return: None
        """
        plt.figure()
        plt.scatter(self.K0_arr, self.loss_arr, color="dodgerblue", marker="x")
        plt.title(f"K0 Minimized: {self.K0}")
        plt.xlabel("K0")
        plt.ylabel("Log Loss")
        plt.show()
        return None

    def _save_mass_radius(self):
        """
        Save mass radius relation for future use through extra calls to TOV integrator.
        :return: None
        """
        max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)
        self.m_R = m_R
        self.r_R = r_R
        return None

    def plot_fluid_perturbations(self):
        """
        Graphical representation of the background solution, fluid perturbations and Lagrangian pressure variations as
        a fucntion of radial distance.
        :return: None
        """
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

    def zerilli_alt(self, r_star, u, omega):
        """
        Alternative version of the Zerilli function as a second order differential equation
        (Converted to first order for scipy).
        :param r_star: Radial distance in tortoise coordinates.
        :param u: Vector used in scipy ode integration.
        :param omega: Exterior oscilaltion frequency multiplied by 2 pi.
        :return: Change in exterior wave solution.
        """
        z, dZdr_star = u
        d2Zdr2_star = z * (self.V_z_alt(r_star, self.m_R, self.nl) - omega * np.conj(omega) / (self.const.c ** 2))
        ret = [dZdr_star, d2Zdr2_star]
        return ret

    def alphas(self, omega, m_R):
        """
        Initial constants as defined by Chandraseakar 1975.
        :param omega: Exterior oscillation frequency multiplied by 2 pi.
        :param m_R: Enclosed mass of the star.
        :return:
        """
        # Define constants and factors.
        n = self.nl
        G = self.const.G
        c = self.const.c

        # Apply the recursion relation to obtain initial values for alpha_i.
        alpha0 = 1 + 1j
        alpha1 = -1j * (n + 1) * alpha0 * c / omega
        alpha2 = alpha0 * (c ** 2) * (-n * (n + 1) + 1j * m_R * omega * (G / (c ** 3)) * (3 / 2 + 3 / n)) / (
                2 * omega ** 2)
        return alpha0, alpha1, alpha2

    def V_z_alt(self, r, m_R, nl):
        """
        Alternate version of the Zerilli potential used in the Zerilli equation.
        :param r: Radial coorinate at a given point (exterior only).
        :param m_R: Enclosed mass of the star.
        :param nl: Factor of order of term in spherical harmonics, l = 2 corresponds to the primary oscillation mode.
        :return: Zerilli potential to be used in the equation.
        """
        # Initialize transition terms.
        b = self._b(r, m_R)
        n = self, nl

        # Compute Zerilli potential.
        fac = (1 - 2 * b)
        num = 2 * (n ** 2) * (n + 1) + 6 * (n ** 2) * b + 18 * n * (b ** 2) + 18 * (b ** 3)
        dem = (r ** 2) * (n + 3 * b) ** 2
        return fac * num / dem

    def r_star_func(self, r, m_R):
        """
        Convert from radial coordinates to tortoise coordinates.
        :param r: Radial coorinate at a given point (exterior only).
        :param m_R: Enclosed mass of the star.
        :return: Tortoise coordinate value of a given radial coordinate.
        """
        G = self.const.G
        c = self.const.c
        return r + 2 * (G / (c ** 2)) * m_R * np.log(abs((r * (c ** 2)) / (2 * G * m_R) - 1))

    def V_z(self, r, m_R):
        """
        Zerilli potential to be used in the Zerilli equation.
        :param r: Radial coorinate at a given point (exterior only).
        :param m_R: Enclosed mass of the star.
        :return: Zerilli potential to be used in the equation.
        """
        nl = self.nl
        G = self.const.G
        c = self.const.c
        G_c2 = G / (c ** 2)
        num = (1 - 2 * G_c2 * m_R / r)
        dem = (r ** 3) * ((nl * r + 3 * G_c2 * m_R) ** 2)
        fac1 = 2 * nl ** 2 * (nl + 1) * (r ** 3)
        fac2 = 6 * (G_c2 ** 1) * (nl ** 2) * m_R * (r ** 2)
        fac3 = 18 * (G_c2 ** 2) * nl * (m_R ** 2) * r
        fac4 = 18 * (G_c2 ** 3) * (m_R ** 3)
        fac = fac1 + fac2 + fac3 + fac4
        ret = fac * num / dem
        return ret

    def zerilli(self, r_star, u, omega):
        """
        Zerilli function as a second order differential equation (Converted to first order for scipy).
        :param r_star: Radial distance in tortoise coordinates.
        :param u: Vector used in scipy ode integration.
        :param omega: Exterior oscilaltion frequency multiplied by 2 pi.
        :return: Change in exterior wave solution.
        """
        omega2 = pow(omega, 2)
        z, dZdr_star = u
        d2Zdr2_star = z * (self.V_z(r_star, self.m_R) - omega2 / (self.const.c ** 2))
        ret = [dZdr_star, d2Zdr2_star]  # dZ/dr*, d2Z/dr*2
        return ret

    def zrly(self, omega, r_star_vals, progress=True):
        """
        Zerilli functional ODE solver.
        :param omega: Exterior oscilaltion frequency multiplied by 2 pi.
        :param r_star_vals: Radial distance in tortoise coordinates as an array.
        :param progress: Should the code display the progress to the user?
        :return: Values for the Zerilli "wave-like" equation and its first derivative.
        """
        # Initialize compelx ODE.
        r = complex_ode(lambda r, VEC: self.zerilli(r, VEC, omega)).set_integrator('LSODA', atol=1.49012e-8,
                                                                                   rtol=1.49012e-8)
        # Set initial values and arrays.
        r.set_initial_value(self.z0, self.r_star_func(self.r_R, np.real(self.m_R)))
        results = [self.z0]
        r_list = [self.r_star_func(self.r_R, self.m_R)]
        i = 0
        if progress:
            self.pbar = tqdm(total=len(r_star_vals))

        # Keep looping while the integration is successful unless a break condition is reached.
        while r.successful():
            i += 1
            if i >= len(r_star_vals):
                break
            if progress:
                self.pbar.update(1)

            integral = r.integrate(r_star_vals[i])
            if not r.successful():
                break

            # Save the data to an array.
            results.append(integral)
            r_list.append(r_star_vals[i])

        # Return the array to be used in future functions. The last term near "infinity" is important.
        results = np.array(results, dtype=complex)
        z, dzdr = results.T
        r = np.array(r_list)
        self.z, self.dzdr, self.r = z, dzdr, r
        return z, dzdr, r

    @staticmethod
    def quadratic(x, a, b, c):
        """
        Quadratic function used to fit any array.
        :param x: x-dependent variable
        :param a: Power term scale factor
        :param b: Linear term scale factor
        :param c: Constant term scale factor.
        :return: Quadratic functional value given inputs.
        """
        return a * (x ** 2) + b * x + c

    def optimize_fmode(self, hw=None, progress=True):
        """
        Given current initial conditions, setup and limits of omega. Find the optimum value for omega (fmode*2pi)
        which reduces the incoming wave solution to zero at infinity. Refer to the methods section of the paper for a
        more detailed explanation about it. :param hw: :param progress: :return:
        """

        # Define constants.
        c = self.const.c
        G = self.const.G

        # Assign imaginary part of the oscillation frequency
        imag = self.imag

        # If half-width is not assigned, assign a broad general width, else use half-width assigned.
        if hw is None:
            fmin = 2.8e3
            fmax = 1.3e3

        else:
            fmin = self.fguess - hw
            fmax = self.fguess + hw

        # Do you want the code to display the progress bar?
        if progress:
            for_loop = tqdm(np.linspace(fmin * 2 * np.pi + imag, fmax * 2 * np.pi + imag, 10))
        else:
            for_loop = np.linspace(fmin * 2 * np.pi + imag, fmax * 2 * np.pi + imag, 10)

        # Initialize empty lists for various fmodes (omega).
        omega_vals = []
        loss_vals = []
        abs_loss_vals = []

        # Loop of every available omega. Find the optimum omega which eliminiates incoming waves at infinity.
        for omega in for_loop:

            # Set initial conditions for first integration
            self.initial_conditions(k=self.main_index_k)

            # Override integration omega and complete first integration.
            self.omega = omega
            self.tov()

            # Update values for integration depending on temporal component of metric.
            self.update_initial_conditions()
            time.sleep(0.1)
            self.tov()
            self.update_initial_conditions()
            time.sleep(0.1)
            self.tov()
            self._save_mass_radius() # Save mass and radius for future use.

            # Extract values at the end of the interior integration to be used as initial values for the Zerilli.
            max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, interior = \
                self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)

            # If the radius is not a reasonable value, skip this step of the integraiton.
            if r_R < 3 * self.const.km2cm:
                print(f"[ERROR] Radius is only {r_R / self.const.km2cm}km. Skipping...")
                continue

            # Apply boudnry minimization step to obtain the optimum value for K0
            self.minimize_K0()
            self.init_VEC = [self.p_c, self.m0, self.nu0, self.H10, self.K0, self.W0, self.X0]
            time.sleep(0.2)
            self.tov()

            # Once the optimum value for K0 has been found, repeat integration and extract surface values.
            max_idx, m_R, r_R, p_R, ec_R, nu_R, h1_R, k_R, w_R, x_R, schild, interior = \
                self._surface_conditions(self.p, self.m, self.r_arr, self.nu, self.h1, self.k, self.w, self.x)

            # Compute initial values for the Zerilli differential equation. Refer to arXiv 2204.03037.
            h1r = h1_R
            Kr = k_R
            b = self._b(r_R, m_R)
            n = self.nl
            r = r_R
            z_R = r * (Kr / (1 - 2 * b) - h1r) / (-(-3 * b ** 2 - 3 * b * n + n) / ((1 - 2 * b) * (3 * b + n)) + (
                    6 * b ** 2 + 3 * b * n + n * (n + 1)) / ((1 - 2 * b) * (3 * b + n)))
            dzdR_R = (-Kr * (-3 * b ** 2 - 3 * b * n + n) / ((1 - 2 * b) * (3 * b + n)) + h1r * (
                    6 * b ** 2 + 3 * b * n + n * (n + 1)) / (3 * b + n)) / (
                             -(-3 * b ** 2 - 3 * b * n + n) / ((1 - 2 * b) * (3 * b + n)) + (
                             6 * b ** 2 + 3 * b * n + n * (n + 1)) / ((1 - 2 * b) * (3 * b + n)))

            self.z0 = [z_R, dzdR_R] # This is the new "INIT VEC"
            alpha0, alpha1, alpha2 = self.alphas(omega, m_R) # Obtain reccursion values for alpha.

            # Integrate from the surface to inf
            r_vals = np.linspace(r_R, 25 * c / np.real(omega), 500)
            r_star_vals = self.r_star_func(r_vals, np.real(m_R))
            z, dzdr_star, r_int = self.zrly(omega, r_star_vals, progress=False) # Solve ODE

            # Save final values at infinity to obtain the incoming and outgoing solutions.
            zFinal = z[-1]
            zPrimeFinal = dzdr_star[-1]
            rFinal = r_vals[-1]
            rStarFinal = r_star_vals[-1]

            # Define transition function at infinity.
            b = self._b(rFinal, m_R)

            # The following steps are all done using 2204.03037, refer to methods for more details.

            # Z+/- at infinity.
            zMinus = np.exp(-1j * (omega / c) * rStarFinal) * (alpha0 + alpha1 / rFinal + alpha2 / (rFinal ** 2))
            zPlus = np.conjugate(zMinus)

            # Zprime +/- at infinity.
            zPrimeMinus = -1j * (omega / c) * np.exp(-1j * (omega / c) * rStarFinal) \
                          * (alpha0 + alpha1 / rFinal + (alpha2 + 1j * alpha1 * (1 - 2 * b) * c / omega) / (
                    rFinal ** 2))
            zPrimePlus = np.conjugate(zPrimeMinus)

            # A+ at infinity.
            A_plus = -zFinal * zPrimeMinus / (zMinus * zPrimePlus - zPlus * zPrimeMinus) \
                     + zMinus * zPrimeFinal / (zMinus * zPrimePlus - zPlus * zPrimeMinus)

            # A- at infinity.
            A_minus = -zFinal * zPrimePlus / (-zMinus * zPrimePlus + zPlus * zPrimeMinus) \
                      + zPlus * zPrimeFinal / (-zMinus * zPrimePlus + zPlus * zPrimeMinus)

            # Define amplitude of incoming wave as the loss, we want this to go to zero at infinity.
            loss = A_plus
            abs_loss = abs(loss)

            # If the loss is wacky, then ignore this step of the for loop.
            if abs_loss > 1e3:
                continue

            # Add loss values for a given omega value
            omega_vals.append(omega)
            loss_vals.append(loss)
            abs_loss_vals.append(abs_loss)

        omega_arr = np.array(omega_vals)
        loss_arr = np.array(loss_vals)
        abs_loss_arr = np.array(abs_loss_vals)
        fmode_arr = np.real(omega_arr / (2 * np.pi))

        # Following 1983ApJS, fit the incoming amplitude with a quadratic function to find the zero crossing.
        # Zero corossing corresponds to the fmode, repeat this step with greater accuracy to obtain more accurate fmode.
        vec = np.real(loss_arr)
        popt, pcov = curve_fit(self.quadratic, xdata=np.real(fmode_arr), ydata=vec)
        f_arr = np.linspace(fmin * 2 * np.pi + imag, fmax * 2 * np.pi + imag, 500) / (2 * np.pi)
        fmode = np.real(f_arr[np.argmin(np.abs(self.quadratic(np.real(f_arr), *popt)))])

        # Analytically compute fmode through quadratic roots.
        _math_a, _math_b, _math_c = popt
        det = np.sqrt(_math_b ** 2 - 4 * _math_a * _math_c)
        fmode1 = (-_math_b + det) / (2 * _math_a)
        fmode2 = (-_math_b - det) / (2 * _math_a)

        # If the roots are close together, then average them.
        if abs(fmode1 - fmode2) < 250:
            fmode = (fmode1 + fmode2) / 2

        # Assign values to class for future reference and analysis.
        self.fguess = fmode
        self.omega_arr = omega_arr
        self.fmode_arr = fmode_arr
        self.abs_loss_arr = abs_loss_arr
        self.loss_arr2 = loss_arr
        self.f_arr = f_arr
        self.popt = popt
        self.fmode = fmode

    def solve_exterior(self, progress=False):
        """
        Repeat the fmode optimization multiple times with decreasing half-width (hw) to find true fundamental mode.
        :param progress: Progress tracker.
        :return: None
        """
        self.optimize_fmode(hw=None, progress=progress)
        self.optimize_fmode(hw=self.hw1, progress=progress)
        self.optimize_fmode(hw=self.hw2, progress=progress)
        self.optimize_fmode(hw=self.hw3, progress=progress)
        return None

    def plot_loss_fmode(self):
        """
        Plot the loss of the fmode for this itegration. Show the true loss with fitted loss and zero crossing.
        :return:
        """
        plt.figure(dpi=100)
        sc = plt.scatter(np.real(self.loss_arr2), np.imag(self.loss_arr2), c=self.fmode_arr, cmap=cm.rainbow)
        plt.xlabel("Re[A+]")
        plt.ylabel("Im[A+]")
        plt.colorbar(sc, label="fmode")
        plt.tight_layout()
        plt.axhline(0)
        plt.axvline(0)
        lim = 20 * min(np.abs(self.loss_arr2))
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.show()

        plt.figure(dpi=100)
        plt.plot(np.real(self.fmode_arr), np.real(self.loss_arr2), label="simulation")
        plt.plot(np.real(self.f_arr), self.quadratic(np.real(self.f_arr), *self.popt), label="fit")
        plt.axvline(self.fmode, label="fmode", color="red")
        plt.title(f"fmode:{self.fmode}")
        plt.xlabel("fmode")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        return None

if __name__ == "__main__":
    test = GeneralRelativity()
