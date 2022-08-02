#!/usr/bin/env python
__author__ = "Utkarsh Mali"
__copyright__ = "Canadian Institute of Theoretical Astrophysics"

from general_relativity.generalRelativity import GeneralRelativity
from general_relativity.__init__ import utkarshGrid
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import re


class fmodes(GeneralRelativity):
    """
    Wrapper child class for oscillation mode, use parallel computing to computer fundamental modes. Computing in the
    fully relatavistic case
    """

    def __init__(self):
        super().__init__()
        pass
