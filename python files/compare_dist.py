# Compare Gamma Distribution to Exponential Distribution when it comes to 
# the catastrophe times of microtubules
from functions import *
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot datashader bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"
# ------------------------------
try:
    import multiprocess
except:
    import multiprocessing as multiprocess

import warnings
    
import numpy as np
import pandas as pd

import bebi103
import iqplot
import scipy
import scipy.stats as st
import holoviews as hv
import holoviews.operation.datashader
hv.extension('bokeh')

import bokeh
bokeh.io.output_notebook()
bebi103.hv.set_defaults()

import numpy.random
rg = numpy.random.default_rng()
import random
import numba
import panel as pn
from scipy.stats import gamma

# Download and tidy the data
df =  pd.read_csv("data/gardner_mt_catastrophe_only_tubulin.csv", header=9)
df12 = pd.DataFrame(df['12 uM'])
df12.columns = ['time']
df7 = pd.DataFrame(df['7 uM'])
df9 = pd.DataFrame(df['9 uM'])
df10 = pd.DataFrame(df['10 uM'])
df14 = pd.DataFrame(df['14 uM'])
con = [12,7,9,10,14]
frames = [df12, df7, df9, df10, df14]

for i, df in enumerate(frames):
    # get name 
    name = df.columns.tolist()
    df.columns = ['time']
    df["concentration"] = con[i]

df = pd.concat(frames)
df = df.dropna()
df = df.sort_values('concentration')
df.head()

rg = np.random.default_rng()


def compare_ecdfs(mle_exp, gamma_mle, times):
    alpha_g = gamma_mle[0]
    beta_g = gamma_mle[1]
    beta1_exp = mle_exp[0]
    beta2_exp = mle_exp[1]
    
    gamma_samples = np.array(
        [rg.gamma(alpha_g, 1 / beta_g, size=len(times)) for _ in range(1000)]
        )

    exp_samples = np.array(
        [rg.exponential(1/beta1_exp, size=len(times)) + rg.exponential(1/beta2_exp, size=len(times)) for _ in range(1000)]
        )
    
    p1_gamma = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, data=times, discrete=True, x_axis_label="n"
    )
    p2_gamma = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, data=times, diff='ecdf', discrete=True, x_axis_label="n"
    )
    p1_exp = bebi103.viz.predictive_ecdf(
    samples=exp_samples, data=times, discrete=True, x_axis_label="n", color="red",
    )
    p2_exp = bebi103.viz.predictive_ecdf(
    samples=exp_samples, data=times, diff='ecdf', discrete=True, x_axis_label="n", color="red",
    )

    return p1_gamma, p2_gamma, p1_exp, p2_exp


def show_ecdfs(con):
    times = df.loc[df['concentration'] == con, 'time']
    gamma_mle = mle_iid_gamma(times)
    mle_exp = mle_iid_exp(times)
    p = compare_ecdfs(mle_exp, gamma_mle, times)
    return p


def ecdf_widgets():
    select_con = pn.widgets.Select(name='Select Concentration', options=['7','9','10','12','14'])
    select_dist = pn.widgets.Select(name='Select Distribution', options=['Exponential', 'Gamma'])
    select_diff = pn.widgets.Select(name='Select Type of Graph', options=['Regular', 'Difference'])
    
    @pn.depends(
        con = select_con.param.value,
        dist = select_dist.param.value, 
        diff = select_diff.param.value
        
    )
    
    def plot_interactive_overlay(con, dist, diff):
        if diff == 'Regular' and dist == 'Gamma':
            return show_ecdfs(int(con))[0]
        elif diff == 'Difference' and dist == 'Gamma':
            return show_ecdfs(int(con))[1]
        if diff == 'Regular' and dist == 'Exponential':
            return show_ecdfs(int(con))[2]
        if diff == 'Difference' and dist == 'Exponential':
            return show_ecdfs(int(con))[3]
    
    widgets = pn.Column(
        pn.Spacer(height=20),
        select_con, 
        pn.Spacer(height=20), 
        select_dist, 
        pn.Spacer(height=20),
        select_diff
    
    )

    row1 = pn.Row(plot_interactive_overlay, pn.Spacer(width=15), widgets)
    return row1

