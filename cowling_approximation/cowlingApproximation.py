#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

import initialize
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import ode
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar


class CowlingApproximation:
    def __init__(self):
        self.e, self.p = None, None
        self.path = None
        self.dedp = None
        self.pbar = None
        self.const = initialize.Constants()
        self.data = initialize.DataManagement()

    def load_path(self, path):
        self.path = path

    def load_e_p(self):
        df = pd.read_csv(self.path)
        self.e, self.p = self.data.df_to_ep(df)

    def load_dedp(self):
        self.dedp = self.dedP(self.p, self.e)

    def lamba_metric(self, M, R):
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

    def _dvdr(self, r, Q, lamda):
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

    def _dUdlnr(self, r, W, U, lamda, l, v):
        return np.exp(lamda / 2 - v) * (W - l * (np.exp(v - lamda / 2)) * U)

    def dedP_helper(self, p, e):
        return np.gradient(e, p), e

    def dedP(self, p, e):
        dedp_helper, e_arr = self.dedP_helper(p, e)
        return interp1d(e_arr, dedp_helper, fill_value="extrapolate", kind="cubic")

    def drhodP(self, e):
        c = self.const.c
        return (c ** -2) * self.dedp(e)

    def coupledTOV(self, r, VEC, init_params):
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
        c_ad2_inv = self.drhodP(e)
        dPdr = self._dPdr(r, P, M, e)
        dMdr = self._dMdr(r, e)
        dvdr = self._dvdr(r, Q, lamda)
        dWdlnr = self._dWdlnr(r, W, U, Q, lamda, l, omega, v, c_ad2_inv)
        dUdlnr = self._dUdlnr(r, W, U, lamda, l, v)
        dWdr = dWdlnr * 1 / r
        dUdr = dUdlnr * 1 / r
        ret = [dPdr, dMdr, dvdr, dWdr, dUdr]
        return ret

    def tov(self, EOS, init_VEC, r_i, p_min, p_max, omega, progress=False,
            l=2, n_iter_max=20000):
        init_params = [EOS, l, omega, p_min, p_max]
        #     r = ode(lambda r, VEC: self.coupledTOV(r, VEC, init_params)).set_integrator('LSODA')
        r = ode(lambda r, VEC: self.coupledTOV(r, VEC, init_params)).set_integrator('VODE')
        r.set_initial_value(init_VEC, r_i)
        results = []
        r_list = []
        i = 0
        r_max = 20 * self.const.km2cm
        max_iter = n_iter_max
        dr = r_max / max_iter
        if progress:
            self.pbar = tqdm(total=max_iter)
        while r.successful() and (r.y[0] >= p_min):
            i += 1
            integral = r.integrate(r.t + dr)
            if progress:
                self.pbar.update(1)
            if i > max_iter:
                print("[STATUS] max_iter reached")
                break
            if r.y[0] < p_min:
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
        return p, m, r, v, w, u

    def initial_conditions(self, EOS, e, p, k, km2cm=1e5, r_i=1):
        G = self.const.G
        c = self.const.c
        p0 = p[k]
        e0 = EOS(p0)
        p_c = p0 - 2 * np.pi * (G / (c ** 4)) * r_i ** 2 * (p0 + e0) * (3 * p0 + e0) / 3
        e_c = EOS(p_c)
        m0 = e_c / (c ** 2) * 4 / 3 * np.pi * r_i ** 3
        omega = 2 * (2 * np.pi)
        l = 2
        v0 = -1
        W0 = 1
        U0 = W0 / (l * np.exp(v0))
        init_VEC = [p_c, m0, v0, W0, U0]
        p_max = max(p)
        p_min = max(c ** 2, min(p))
        return km2cm, r_i, p0, e0, p_c, e_c, m0, omega, l, v0, W0, U0, init_VEC, p_min, p_max

    def surface_conditions(self, p, m, r_arr, v, w, u):
        G = self.const.G
        c = self.const.c
        max_idx = np.argmax(m) - 1
        m_R = m.max()  # In units of msun
        r_R = r_arr[max_idx]  # In units of km
        p_R = p[max_idx]  # cgs
        ec_R = EOS(p_R)  # cgs
        u_R = u[max_idx]  # cgs
        v_R = v[max_idx]
        w_R = w[max_idx]
        schild = (1 - 2 * G * m_R / (c ** 2 * r_R))
        interior = np.exp(v_R)
        return max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, schild, interior

    def boundary_wu(self, r_R, m_R, omega, w_R, u_R):
        G = self.const.G
        c = self.const.c
        frac1 = (omega ** 2 * r_R ** 3) / (G * m_R)
        return frac1 * np.sqrt(1 - (2 * G * m_R) / (r_R * (c ** 2))) - w_R / u_R

    def print_params(self, p, m, r_arr, v, w, u):
        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self.surface_conditions(p, m, r_arr, v, w, u)
        print(f"Star has mass {m_R / self.const.msun:.3f} Msun and radius {r_R / self.const.km2cm:.3f}km")
        print(f"Interior Surface: {interior:.8f}")
        print(f"Exterior Surface: {schild:.8f}")
        print(f"v0: {v0}")
        print(f"Lamda: {self.lamba_metric(m_R, r_R)}")
        print(f"Boundary Term: {self.boundary_wu(r_R, m_R, omega, w_R, u_R)}")
        return None

    def minimize_boundary(self, params, p=p, EOS=EOS):
        # Repeat integration
        omega = params

        # Integrate
        p, m, r_arr, v, w, u = self.tov(EOS, init_VEC, r_i, p_min, p_max, omega, l=l)

        max_idx, m_R, r_R, p_R, ec_R, u_R, v_R, w_R, \
        schild, interior = self.surface_conditions(p, m, r_arr, v, w, u)

        loss = np.log10(abs(self.boundary_wu(r_R, m_R, omega, w_R, u_R)))
        return loss

    def get_omega_bounds(self, mass_arr, radius_arr):
        lower = 2 * np.pi * (0.60e3 + 23e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        upper = 2 * np.pi * (0.8e3 + 50e-6 * np.sqrt(mass_arr / (radius_arr ** 3)))
        return lower, upper

    def optimize_fmode(self, func, m_R, r_R, newt_approx=False):
        if newt_approx:
            omega_min, omega_max = self.get_omega_bounds(m_R, r_R)
            res = minimize_scalar(func,
                                  bounds=(omega_min, omega_max),
                                  method='bounded',
                                  options={"maxiter": 30})
            omg = res.x

        else:
            omega_guess = (2 * np.pi) * (0.70e3 + 30e-6 * np.sqrt(m_R / (r_R ** 3)))
            init_guess = [omega_guess]
            res = minimize(func, x0=init_guess, method='Nelder-Mead',
                           options={"disp": True, "maxiter": 15})
            omg = res.x[0]

        f = omg / (2 * np.pi)
        return res, f


if __name__ == "__main__":
    experiment = CowlingApproximation()
