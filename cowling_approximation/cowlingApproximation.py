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
import time


class CowlingApproximation:
    def __init__(self):
        self.e, self.p, self.EOS, self.path, self.dedp, self.pbar, self.v0 = None, None, None, None, None, None, None
        self.W0, self.U0, self.init_VEC, self.p_max, self.p_min, self.p_max = None, None, None, None, None, None
        self.p0, self.e0, self.p_c, self.e_c, self.m0, self.omega, self.l = None, None, None, None, None, None, None
        self.r_i, self.m, self.v, self.w, self.u, self.r_arr, self.e_arr = None, None, None, None, None, None, None
        self.p_arr, self.m_R, self.r_R, self.f, self.res = None, None, None, None, None
        self.n_iter_max = 20000
        self.const = initialize.Constants()
        self.data = initialize.DataManagement()

    @staticmethod
    def _get_ep(e, p):
        f_e_smooth = interp1d(p, e, fill_value="extrapolate", kind="cubic")
        return f_e_smooth

    @staticmethod
    def _get_pe(p, e):
        f_e_smooth = interp1d(e, p, fill_value=(0, 0), kind="cubic", bounds_error=True)
        return f_e_smooth

    def read_data(self, path):
        self.path = path
        df = pd.read_csv(path)
        e, p = self.data.df_to_ep(df)
        EOS = self._get_ep(e, p)
        self.e_arr = e
        self.p_arr = p
        self.EOS = EOS
        return None

    def load_dedp(self):
        self.dedp = self._dedP(self.p_arr, self.e_arr)

    @staticmethod
    def _dedP_helper(p_arr, e_arr):
        return np.gradient(e_arr, p_arr), e_arr

    def _dedP(self, p_arr, e_arr):
        dedp_helper, e_arr = self._dedP_helper(p_arr, e_arr)
        return interp1d(e_arr, dedp_helper, fill_value="extrapolate", kind="cubic")

    def _drhodP(self, e):
        c = self.const.c
        return (c ** -2) * self.dedp(e)

    def _lamda_metric(self, M, R):
        G = self.const.G
        c = self.const.c
        return -np.log((1 - 2 * G * M / (c ** 2 * R)))

    def _dMdr(self, r, e):
        c = self.const.c
        return 4 * np.pi * r ** 2 * (e / (c ** 2))

    def _b(self, r, M):
        G = self.const.G
        c = self.const.c
        return (G * M) / ((c ** 2) * r)

    @staticmethod
    def _dvdr(r, Q, lamda):
        return (2 / r) * np.exp(lamda) * Q

    def _Q(self, r, P, M):
        G = self.const.G
        c = self.const.c
        frac = (4 * np.pi * G * (r ** 2) * P) / (c ** 4)
        return self._b(r, M) + frac

    def _dPdr(self, r, P, M, e):
        G = self.const.G
        c = self.const.c
        num = (M + 4 * np.pi * (r ** 3) * P / (c ** 2))
        dem = r * (r - 2 * G * M / (c ** 2))
        return -1 * (e + P) * G / (c ** 2) * num / dem

    def _dWdlnr(self, r, W, U, Q, lamda, l, omega, v, c_ad2_inv):
        c = self.const.c
        term1 = -1 * (l + 1) * (W - l * np.exp(v + lamda / 2) * U)
        frac = -1 * ((omega * r) ** 2) * (np.exp(lamda / 2) * c_ad2_inv)
        term2 = frac * (U - np.exp(lamda / 2) * Q * W * (c ** 2) / ((omega * r) ** 2))
        return term1 + term2

    @staticmethod
    def _dUdlnr(W, U, lamda, l, v):
        return np.exp(lamda / 2 - v) * (W - l * (np.exp(v - lamda / 2)) * U)

    def _coupledTOV(self, r, VEC, init_params):
        P, M, v, W, U = VEC
        EOS, l, omega, p_min, p_max = init_params
        if P <= p_min:
            return None
        if P >= p_max:
            return None
        if 2 * self._b(r, M) >= 1:
            return None

        lamda = np.log(1 / (1 - 2 * self._b(r, M)))
        Q = self._Q(r, P, M)
        e = EOS(P)
        c_ad2_inv = self._drhodP(e)
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
        self.load_dedp()
        G = self.const.G
        c = self.const.c
        self.r_i = 1
        self.p0 = self.p_arr[k]
        self.e0 = self.EOS(self.p0)
        self.p_c = self.p0 - 2 * np.pi * (G / (c ** 4)) * self.r_i ** 2 * (self.p0 + self.e0) * \
                   (3 * self.p0 + self.e0) / 3
        self.e_c = self.EOS(self.p_c)
        self.m0 = self.e_c / (c ** 2) * 4 / 3 * np.pi * self.r_i ** 3
        self.omega = 2 * (2 * np.pi)
        self.l = 2
        self.v0 = -1
        self.W0 = 1
        self.U0 = self.W0 / (self.l * np.exp(self.v0))
        self.init_VEC = [self.p_c, self.m0, self.v0, self.W0, self.U0]
        self.p_max = max(self.p_arr)
        self.p_min = max(c ** 2, min(self.p_arr))
        return self.const.km2cm, self.r_i, self.p0, self.e0, self.p_c, self.e_c, self.m0, self.omega, self.l, \
               self.v0, self.W0, self.U0, self.init_VEC, self.p_min, self.p_max

    def tov(self, progress=False):
        init_params = [self.EOS, self.l, self.omega, self.p_min, self.p_max]
        r = ode(lambda _r, VEC: self._coupledTOV(_r, VEC, init_params)).set_integrator('VODE')
        r.set_initial_value(self.init_VEC, self.r_i)
        results = []
        r_list = []
        i = 0
        r_max = 20 * self.const.km2cm
        max_iter = self.n_iter_max
        dr = r_max / max_iter
        if progress:
            self.pbar = tqdm(total=max_iter)
        while r.successful() and (r.y[0] >= self.p_min):
            i += 1
            integral = r.integrate(r.t + dr)
            if progress:
                self.pbar.update(1)
            if i > max_iter:
                print("[STATUS] max_iter reached")
                break
            if r.y[0] < self.p_min:
                break
            if not r.successful():
                break
            results.append(integral)
            r_list.append(r.t + dr)
        if progress:
            self.pbar.close()

        results = np.array(results, dtype=float)
        p, m, v, w, u = results.T
        r = np.array(r_list)
        self.p, self.m, self.v, self.w, self.u, self.r_arr = p, m, v, w, u, r
        return p, m, r, v, w, u

    def update_initial_conditions(self):
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, \
        interior = self._surface_conditions(self.p, self.m, self.r_arr, self.v, self.w, self.u)
        v_ext = -self._lamda_metric(m_R, r_R)
        v_int = v_R  # At surface
        delta_v = v_int - v_ext
        self.v0 = self.v0 - delta_v
        self.U0 = self.W0 / (self.l * np.exp(self.v0))
        self.init_VEC = [self.p_c, self.m0, self.v0, self.W0, self.U0]
        return None

    def _surface_conditions(self, p, m, r_arr, v, w, u):
        G = self.const.G
        c = self.const.c
        max_idx = np.argmax(m) - 1
        m_R = m.max()  # In units of msun
        r_R = r_arr[max_idx]  # In units of km
        p_R = p[max_idx]  # cgs
        ec_R = self.EOS(p_R)  # cgs
        u_R = u[max_idx]  # cgs
        v_R = v[max_idx]
        w_R = w[max_idx]
        schild = (1 - 2 * G * m_R / (c ** 2 * r_R))
        interior = np.exp(v_R)
        return max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, interior

    def _boundary_wu(self, r_R, m_R, omega, w_R, u_R):
        G = self.const.G
        c = self.const.c
        frac1 = (omega ** 2 * r_R ** 3) / (G * m_R)
        return frac1 * np.sqrt(1 - (2 * G * m_R) / (r_R * (c ** 2))) - w_R / u_R

    def print_params(self):
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
        p, m, r_arr, v, w, u = self.tov()
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self._surface_conditions(p, m, r_arr, v, w, u)
        loss = np.log10(abs(self._boundary_wu(r_R, m_R, omega, w_R, u_R)))
        return loss

    def _save_mass_radius(self):
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self._surface_conditions(self.p, self.m, self.r_arr, self.v, self.w, self.u)
        self.m_R = m_R
        self.r_R = r_R
        return None

    @staticmethod
    def get_omega_bounds(mass_arr, radius_arr):
        lower = 2 * np.pi * (0.60e3 + 23e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        upper = 2 * np.pi * (0.8e3 + 50e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        return lower, upper

    def optimize_fmode(self, newt_approx=True):
        self._save_mass_radius()
        if newt_approx:
            omega_min, omega_max = self.get_omega_bounds(self.m_R, self.r_R)
            res = minimize_scalar(self._minimize_boundary,
                                  bounds=(omega_min, omega_max),
                                  method='bounded',
                                  options={"maxiter": 30})
            omg = res.x

        else:
            omega_guess = (2 * np.pi) * (0.70e3 + 30e-6 * np.sqrt(self.m_R / (self.r_R ** 3)))
            init_guess = [omega_guess]
            res = minimize(self._minimize_boundary, x0=init_guess, method='Nelder-Mead',
                           options={"disp": True, "maxiter": 15})
            omg = res.x[0]

        f = omg / (2 * np.pi)
        self.f = f
        self.res = res
        return None


if __name__ == "__main__":
    test = CowlingApproximation()
