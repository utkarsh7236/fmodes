#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"


class Constants:
    """
    Defining constants used across all codes.
    """
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
    """
    Data Wrangling and Management
    """
    def __init__(self):
        self.const = Constants()

    @staticmethod
    def trim_ep(e_arr, p_arr):
        p = p_arr[p_arr > 1e8]
        e = e_arr[p_arr > 1e8]
        return e, p

    def df_to_ep(self, df):
        """
        Converts df to readable pressure and energy density in CGS units.
        :param df: pandas dataframe containing pressurec2, and energy_densityc2 columns in units of 1/c^2
        :return: pressure and energy density in CGS units
        """
        c = self.const.c
        _e_den = df.energy_densityc2
        _p = df.pressurec2
        e_den, p = self.trim_ep(_e_den, _p)
        e_den_normed = e_den * (c ** 2)
        pressure = p * (c ** 2)
        return e_den_normed.to_numpy(), pressure.to_numpy()