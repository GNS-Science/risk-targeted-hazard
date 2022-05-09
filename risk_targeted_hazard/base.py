import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import json
import glob
import re
import h5py

from pathlib import Path
from openquake.commonlib import datastore
from scipy import stats
from scipy.optimize import minimize


# set all single line variables to be displayed, not just the last line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

g = 9.80665 # gravity in m/s^2

def acc_to_disp(acc,t):
    return (acc * g) * (t/(2*np.pi))**2


def disp_to_acc(disp,t):
    return disp / (t/(2*np.pi))**2 / g


def sigfig(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def find_nearest(a0, a):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a[idx]


def period_from_imt(imt):
    if imt in ['PGA','PGD']:
        period = 0
    else:
        period = float(re.split('\(|\)',imt)[1])
    return period

def imt_from_period(period):
    if period == 0:
        imt = 'PGA'
    else:
        imt = f'SA({period})'
    return imt


def find_fragility_median(im_value, beta, design_prob):
    return minimize(median_based_on_p_collapse, im_value, args=(im_value, beta, design_prob), method='Nelder-Mead').x[0]


def median_based_on_p_collapse(x, im, beta, target_prob):
    return np.abs(target_prob - stats.lognorm(beta, scale=x).cdf(im))[0]


def set_plot_formatting():
    # set up plot formatting
    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 25

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


set_plot_formatting()
