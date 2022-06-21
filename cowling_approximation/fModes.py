#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

from joblib import Parallel, delayed
from cowling_approximation.cowlingApproximation import CowlingApproximation
from tqdm import tqdm


class fmodes(CowlingApproximation):
    def __init__(self):
        super(fmodes, self).__init__()
        self.cowling = CowlingApproximation()
        self.results, self.path = None, None

    def set_EOS(self, path):
        self.path = path

    def run_fmode_single(self, k):
        self.cowling = CowlingApproximation()
        self.cowling.read_data(self.path)
        self.cowling.initial_conditions(k=-k)
        self.cowling.tov(progress=True)
        self.cowling.update_initial_conditions()
        self.cowling.tov(progress=True)
        self.cowling.optimize_fmode()
        return self.cowling.f, self.m_R, self.r_R

    def process(self, k):
        f, m_R, r_R = self.run_fmode_single(k)
        return f, m_R, r_R, k

    def parallel_simulation(self):
        self.results = Parallel(n_jobs=-2, verbose=0, max_nbytes='8M')(delayed(self.process)(k) for k in tqdm(vals))
