#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

from scipy.interpolate import interp1d
import pandas as pd


class Constants:
    def __init__(self):
        """
        Define constants used in integration code.
        """
        self.G = 6.67259e-8
        self.c = 2.99792458e10
        self.nden = 2.3e14
        self.km2cm = 1e5
        self.msun = 1.98847e33


class DataManagement:
    def __init__(self):
        self.const = initialize.Constants()

    def df_to_ep(self, df):
        c = self.const.c
        e_den = df.energy_densityc2
        e_den_normed = e_den
        p = df.pressurec2
        e_den_normed = e_den_normed * (c ** 2)
        pressure = p * (c ** 2)
        return e_den_normed.to_numpy(), pressure.to_numpy()

    def get_ep(self, e, p):
        f_e_smooth = interp1d(p, e, fill_value="extrapolate", kind="cubic")
        return f_e_smooth

    def get_pe(self, p, e):
        f_e_smooth = interp1d(e, p, fill_value=(0, 0), kind="cubic", bounds_error=True)
        return f_e_smooth

    def read_data(self, path):
        df = pd.read_csv(path)
        e, p = self.df_to_ep(df)
        EOS = self.get_ep(e, p)
        return e, p, EOS
