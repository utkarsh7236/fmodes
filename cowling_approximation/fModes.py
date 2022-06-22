#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

from cowling_approximation.cowlingApproximation import CowlingApproximation
from cowling_approximation.__init__ import utkarshGrid
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import re


class fmodes(CowlingApproximation):
    """
    Wrapper child class for oscillation mode, use parallel computing to computer fundamental modes.
    """

    def __init__(self):
        """
        Initialize, call parent class in Cowling approximation.
        """
        super(fmodes, self).__init__()
        self.cowling = CowlingApproximation()
        self.results, self.path, self.ind_start, self.ind_stop, self.vals = None, None, None, None, None
        self.idx_arr, self.max_idx, self.solar_idx, self.max_idx_arr, self.mass_arr = None, None, None, None, None
        self.f_mode_arr, self.radius_arr, self.max_idx_new, self.EOS_name = None, None, None, None

        # Offload this somewhere, dictionary limiting integration bounds
        self.dic = {"SLY230A.csv": [-2, -200, 5],
                    "SLY4.csv": [-2, -200, 5],
                    "NL3.csv": [-2, -1200, 40],
                    "sly.csv": [-2, -35, 1],
                    "sly_short.csv": [-2, -35, 1],
                    "sly230a.csv": [-2, -120, 2],
                    "nl3cr.csv": [-2, -1140, 20]}

        self.jump = 5  # Jump factor over the EOS values.
        self.hz2khz = 1e-3

    def set_EOS(self, path):
        """
        Use parent Cowling class to read EOS, define integration values and limits.
        :param path: Path to EOS table.
        :return: None
        """
        self.path = path
        self.cowling.read_data(self.path)
        self.e_arr, self.p_arr = self.cowling.e_arr, self.cowling.p_arr
        self.EOS_name = re.search("([^\/]+$)", self.path).group(0)  # Regex search limiting pressure table bounds
        self._get_vals()
        return None

    def _get_vals(self):
        """
        Set limits on EOS table to ignore redundant integrations. (Only integrate useful M/R/Fmodes)
        :return: None
        """

        if self.EOS_name in self.dic:
            self.ind_start, self.ind_stop, self.jump = self.dic[self.EOS_name]
            self.vals = range(self.ind_stop, self.ind_start + 1, 1)[::-self.jump]
        elif self.EOS_name[:4] == "eos_":
            self.ind_start, self.ind_stop, self.jump = -2, -300, 3
            self.vals = range(self.ind_stop, self.ind_start + 1, 1)[::-self.jump]
        else:
            self.ind_start, self.ind_stop = -2, -len(self.p_arr)
            self.vals = range(self.ind_stop, self.ind_start + 1, 1)[::-self.jump]
        return None

    def process(self, k):
        """
        Define wrapper function to integrate. Define new Cowling Approximation class to prevent parallel memory issues.
        :param k: Index of integration dependent on _get_vals()
        :return: oscillation mode, mass at surface, radius at surface, index.
        """
        curr = CowlingApproximation()
        curr.read_data(self.path)
        curr.initial_conditions(k=k)
        curr.tov()
        curr.update_initial_conditions()
        curr.tov()
        curr.optimize_fmode()
        return curr.f, curr.m_R, curr.r_R, k

    def parallel_simulation(self, progress = False):
        """
        Run parallel simulation and save the results. Use all but 1 CPU core.
        :return: None
        """
        if progress:
            self.results = Parallel(n_jobs=-2, verbose=0, max_nbytes='8M') \
                (delayed(self.process)(k) for k in tqdm(self.vals))
        else:
            self.results = Parallel(n_jobs=-2, verbose=0, max_nbytes='8M') \
                (delayed(self.process)(k) for k in self.vals)
        self.mass_arr = np.array(self.results).T[1]
        self.f_mode_arr = np.array(np.array(self.results).T[0])
        self.radius_arr = np.array(self.results).T[2]
        self.idx_arr = np.array(self.results).T[3]

        # Parse
        self.max_idx = self.mass_arr.argmax()
        self.solar_idx = self.idx_arr[(np.abs(self.mass_arr / self.const.msun - 1.4)).argmin()]
        self.max_idx_arr = self.idx_arr[self.max_idx]
        self.mass_arr = self.mass_arr[self.max_idx:]
        self.f_mode_arr = self.f_mode_arr[self.max_idx:]
        self.radius_arr = self.radius_arr[self.max_idx:]
        self.max_idx_new = self.mass_arr.argmax()

    def print_results(self):
        """
        Display results of integration in understandable units.
        :return: None
        """
        print(f"M_max = {self.mass_arr[self.max_idx_new] / self.const.msun}")
        print(f"R_max = {self.radius_arr[self.max_idx_new] / self.const.km2cm}")
        print(f"f_max = {self.f_mode_arr[self.max_idx_new]}")
        print()
        solar_idx = (np.abs(self.mass_arr / self.const.msun - 1.4)).argmin()  # Solar index
        print(f"M_1.4 = {self.mass_arr[solar_idx] / self.const.msun}")
        print(f"R_1.4 = {self.radius_arr[solar_idx] / self.const.km2cm}")
        print(f"f_1.4 = {self.f_mode_arr[solar_idx]}")
        return None

    def plot_fmass(self):
        """
        Plot fundamental mode vs mass relationship.
        :return: None
        """
        plt.figure(dpi=300)
        plt.tight_layout()
        plt.scatter(self.mass_arr / self.const.msun, self.f_mode_arr * self.hz2khz,
                    c=self.radius_arr / self.const.km2cm, marker="x",
                    cmap="plasma")

        # Plot lower and upper limit on omega bounds.
        lower, upper = self.cowling.get_omega_bounds(self.mass_arr, self.radius_arr)
        lower, upper = lower / (1e3 * 2 * np.pi), upper / (1e3 * 2 * np.pi)
        plt.gca().fill_between(self.mass_arr / self.const.msun, lower, upper, alpha=0.3, label="Optimization Bounds")
        plt.xlabel("Mass/Msun")
        plt.ylabel("fmode (kHz)")
        cbar = plt.colorbar()
        cbar.set_label('Radius (km)', rotation=-90, labelpad=15)
        utkarshGrid()
        plt.legend()
        plt.show()

    def plot_mass_radius(self):
        """
        Plot mass vs radius relationship.
        :return: None
        """
        plt.figure(dpi=300)
        plt.plot(self.radius_arr / self.const.km2cm, self.mass_arr / self.const.msun)
        plt.xlabel("Radius (km)")
        plt.ylabel("Mass (Msun)")
        plt.show()

    def plot_fmode_linear(self):
        """
        Plot fmode vs sqrt(Mass/Radius^3) relationship. Should be approximately linear in ideal cases.
        :return: None
        """
        plt.figure(dpi=300)
        plt.plot(np.sqrt((self.mass_arr / self.const.msun) / ((self.radius_arr / self.const.km2cm) ** 3)),
                 self.f_mode_arr * self.hz2khz)
        plt.xlabel("√M/R^3")
        plt.ylabel("fmode (kHz)")
        plt.xlim(0.02, 0.05)
        plt.ylim(1.4, 3)
        plt.show()
        return None
