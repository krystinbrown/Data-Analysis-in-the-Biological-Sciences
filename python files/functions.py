# Colab setup ------------------
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

data =  pd.read_csv("data/gardner_time_to_catastrophe_dic_tidy.csv")


# Code From hw 2.2
def ecdf_vals(data):
    ''' Takes in a Numpy array or Pandas Series and will return 
        x and y values for plotting an ecdf'''
    data = np.sort(data)
    return np.asarray([[val, (i+1)/(len(data))] for i, val in enumerate(data) 
                       if i == len(data)-1 or val != data[i+1]])

def plot_strip_ecdf():
	df =  pd.read_csv("data/gardner_time_to_catastrophe_dic_tidy.csv")
	label = df.loc[df["labeled"], "time to catastrophe (s)"]
	no_label = df.loc[~df["labeled"], "time to catastrophe (s)"]
	p = bokeh.plotting.figure(
    width=700,
    height=500,
    x_axis_label="Catastrophe Time. (s)",
    y_axis_label="Cumulative Distribution",
    title="eCDF of Catastrophe times"
	)

	# prepare list of colors

	colors = ["red", "blue"]
	legend_name = ["labeled", "non-labeled"]
	for i, lab in enumerate([ecdf_vals(label), ecdf_vals(no_label)]):
	    p.circle(
	        x=lab[:,0],
	        y=lab[:,1],
	        color=colors[i],
	        legend_label= legend_name[i]
	    )

	p1 = iqplot.strip(
	    data=df,
	    x_axis_label = 'Time (s)',
	    y_axis_label = 'Labeled?',
	    width=500,
	    height=500,
	    q='time to catastrophe (s)',
	    cats = 'labeled',
	    jitter=True,
        title="Labeling of tubulin",
	)


	p.legend.location = 'bottom_right'
	p.legend.click_policy = "hide"
	bokeh.io.show(p1)
	bokeh.io.show(p)
	return



# Code from HW 6.1
# functions for performing bootstrap 
@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


@numba.njit
def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out

def ecdf(x, data):
    '''Returns the ecdf evaluated at x for data.'''
    data = np.sort(data)
    return numpy.where(data <= x)[0].size/data.size

def run_hw6():
	data =  pd.read_csv("data/gardner_time_to_catastrophe_dic_tidy.csv")
	d_t = data[data["labeled"] == True]
	d_f = data[data["labeled"] == False]
	return

def plot_ecdf_conf(data):
    p = iqplot.ecdf(
        data = data,
        q = "time to catastrophe (s)",
        cats = ["labeled"],
        conf_int = True)
    bokeh.io.show(p)
    
# Construct summaries for confidence interval plot
def plot_conf_int(data):
    '''Construct 95% Confidence Intervals and Plot'''
    from bokeh.plotting import figure
    
    d_t = data[data["labeled"] == True]
    d_f = data[data["labeled"] == False]
    
    label = np.array(d_t["time to catastrophe (s)"])
    no_label = np.array(d_f["time to catastrophe (s)"])
    
    # Labeled
    bs_reps_mean = draw_bs_reps_mean(np.array(d_t["time to catastrophe (s)"]), size=10000)
    # 95% confidence intervals
    mean_conf_int_labeled = np.percentile(bs_reps_mean, [2.5, 97.5])
    
    # Unlabeled
    bs_reps_mean = draw_bs_reps_mean(np.array(d_f["time to catastrophe (s)"]), size=10000)
    # 95% confidence intervals
    mean_conf_int_unlabeled = np.percentile(bs_reps_mean, [2.5, 97.5])
    
    summaries = [
        dict(estimate=np.mean(label), conf_int=mean_conf_int_labeled, label="labeled"),
        dict(
            estimate=np.mean(no_label), conf_int=mean_conf_int_unlabeled, label="unlabeled"
        ),
    ]
    
    bokeh.io.show(bebi103.viz.confints(summaries))

def conf_ints_labeled(data):
	'''Compute the confidence intervals for the labeled and unlabeled tubulin
	   and print them out. Uses nonparametric bootstrapping '''
	# read in data 
	d_t = data[data["labeled"] == True]
	d_f = data[data["labeled"] == False]
	bs_reps_mean = draw_bs_reps_mean(np.array(d_t["time to catastrophe (s)"]), size=10000)

	# 95% confidence intervals
	mean_conf_int = np.percentile(bs_reps_mean, [2.5, 97.5])

	print("""Labeled:
	Mean 95% conf time to catastrophe (s):   [{0:.2f}, {1:.2f}]
	""".format(*(mean_conf_int)))

	bs_reps_mean = draw_bs_reps_mean(np.array(d_f["time to catastrophe (s)"]), size=10000)

	# 95% confidence intervals
	mean_conf_int = np.percentile(bs_reps_mean, [2.5, 97.5])

	print("""Unlabeled: 
	Mean 95% conf time to catastrophe (s):   [{0:.2f}, {1:.2f}]
	""".format(*(mean_conf_int)))
	return 

def find_pval(data):
	d_t = data[data["labeled"] == True]
	d_f = data[data["labeled"] == False]
	label = np.array(d_t["time to catastrophe (s)"])
	no_label = np.array(d_f["time to catastrophe (s)"])
	m = label.size
	n = no_label.size
	# Compute test statistic for original data set
	diff_mean = np.mean(label) - np.mean(no_label)

	# Draw replicates
	perm_reps = draw_perm_reps_diff_mean(label, no_label, size=10000)

	# Compute p-value
	p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)

	print('p-value =', p_val)
	return 

def plot_conf_int_theor(data):
    '''Construct 95% Confidence Intervals and Plot'''
    d_t = data[data["labeled"] == True]
    d_f = data[data["labeled"] == False]
    
    label = np.array(d_t["time to catastrophe (s)"])
    no_label = np.array(d_f["time to catastrophe (s)"])

    mean_conf_int = []
    dat = [label, no_label]
    for i in range(2):
        X = dat[i]
        n = X.size
        mean = np.mean(X)
        var = 1/(n*(n-1))*np.sum((X-mean)**2)

        mean_conf_int += [st.norm.interval(0.95, loc=mean, scale=var**.5)]
    
    summaries = [
        dict(estimate=np.mean(label), conf_int=mean_conf_int[0], label="labeled"),
        dict(
            estimate=np.mean(no_label), conf_int=mean_conf_int[1], label="unlabeled"
        ),
    ]
    
    bokeh.io.show(bebi103.viz.confints(summaries))

def lab_conf_int(data):
	d_t = data[data["labeled"] == True]
	d_f = data[data["labeled"] == False]
	label = np.array(d_t["time to catastrophe (s)"])
	no_label = np.array(d_f["time to catastrophe (s)"])
	name = ["label", "no label"]
	dat = [label, no_label]
	for i in range(2):
	    X = dat[i]
	    n = X.size
	    mean = np.mean(X)
	    var = 1/(n*(n-1))*np.sum((X-mean)**2)
	    
	    conf_int = st.norm.interval(0.95, loc=mean, scale=var**.5)
	    print("{} confidence interval: {}\n".format(name[i], conf_int))
	return



def ecdf_bounds(data):
	p = iqplot.ecdf(
    data = data,
    q = "time to catastrophe (s)",
    cats = ["labeled"],
    conf_int = True
	)
	d_t = data[data["labeled"] == True]
	d_f = data[data["labeled"] == False]
	label = np.array(d_t["time to catastrophe (s)"])
	no_label = np.array(d_f["time to catastrophe (s)"])
	X = np.linspace(0, 2000, 200)
	a = .05
	n = label.size
	eps = np.sqrt((1/(2*n)*np.log(2/a)))

	y_min = np.array([max(0, ecdf(x,label) - eps) for x in X])
	y_max = np.array([min(1, ecdf(x,label) + eps) for x in X])
	p.line(x = X, y = y_min)
	p.line(x = X, y = y_max)

	n = no_label.size
	y_min = np.array([max(0, ecdf(x,no_label) - eps) for x in X])
	y_max = np.array([min(1, ecdf(x,no_label) + eps) for x in X])
	p.line(x = X, y = y_min, color = "orange")
	p.line(x = X, y = y_max, color = "orange")

	bokeh.io.show(p)
	return  

def ecdf_bounds_labeled(data):
    d_t = data[data["labeled"] == True]
    
    p = iqplot.ecdf(
        data = d_t,
        q = "time to catastrophe (s)",
        cats = ["labeled"],
        conf_int = True,
        title="Labeled tubulin"
        )
    
    label = np.array(d_t["time to catastrophe (s)"])
    
    X = np.linspace(0, 2000, 200)
    a = .05
    n = label.size
    eps = np.sqrt((1/(2*n)*np.log(2/a)))

    y_min = np.array([max(0, ecdf(x,label) - eps) for x in X])
    y_max = np.array([min(1, ecdf(x,label) + eps) for x in X])
    p.line(x = X, y = y_min)
    p.line(x = X, y = y_max)

    bokeh.io.show(p)
    return  

def ecdf_bounds_unlabeled(data):
    d_f = data[data["labeled"] == False]
    
    p = iqplot.ecdf(
        data = d_f,
        q = "time to catastrophe (s)",
        cats = ["labeled"],
        conf_int = True,
        title="Unlabeled tubulin",
        palette=["orange"],
    )
    
    no_label = np.array(d_f["time to catastrophe (s)"])
    X = np.linspace(0, 2000, 200)
    a = .05
    n = no_label.size
    eps = np.sqrt((1/(2*n)*np.log(2/a)))

    y_min = np.array([max(0, ecdf(x,no_label) - eps) for x in X])
    y_max = np.array([min(1, ecdf(x,no_label) + eps) for x in X])
    p.line(x = X, y = y_min, color = "orange")
    p.line(x = X, y = y_max, color = "orange")

    bokeh.io.show(p)
    return   


# Code from 8.2
def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. NBinom measurements, parametrized
    by alpha, beta."""
    alpha, beta = params

    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, loc = 0, scale = 1/beta))


def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([1, 1/2]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def plot_ecdf_gamma(data):
    
    t = data.loc[data['labeled'], 'time to catastrophe (s)'].values 
    mle = mle_iid_gamma(t)
    
    bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
        mle_iid_gamma,
        gen_fun_gamma,
        t,
        gen_args=(t, ),
        size=1000,
        n_jobs=3,
        progress_bar=True,
    )

    conf = np.percentile(bs_reps, [2.5, 97.5], axis=0)
    
    p = iqplot.ecdf(t, q='t (s)', conf_int=True)

    t_theor = np.linspace(0, 2000, 200)
    cdf = st.gamma.cdf(t_theor, mle[0], loc=0, scale=1/mle[1])
    p.line(t_theor, cdf, line_width=2, color='orange')

    bokeh.io.show(p)
    
    print('''alpha
        MLE: {}
        95% Confidence Interval {}
          '''.format(mle[0], conf[:, 0]))
    print('''beta
        MLE: {}
        95% Confidence Interval {}
              '''.format(mle[1], conf[:, 1]))

def gen_fun_gamma(params, n, size, rg):
    alpha, beta = params
    return st.gamma.rvs(alpha, loc = 0, scale = 1/beta, size = size)

labeled_time_data = data[data["labeled"] == True]
labeled_time_data = data["time to catastrophe (s)"].values

def gamma_bootstrap(data):
	bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
    mle_iid_gamma,
    gen_fun_gamma,
    data,
    gen_args=(data, ),
    size=1000,
    n_jobs=3,
    progress_bar=False,
	)
	return bs_reps

def gamma_conf(data):
	bs_reps = gamma_bootstrap(data)
	return np.percentile(bs_reps, [2.5,97.5], axis = 0)


def gamma_dist(data):
	conf = gamma_conf(data)
	mle = mle_iid_gamma(data)
	print('''alpha
      MLE: {}
      95% Confidence Interval {}
          '''.format(mle[0], conf[:, 0]))
	print('''beta
      MLE: {}
      95% Confidence Interval {}
          '''.format(mle[1], conf[:, 1]))
	return


def log_like_iid_exp(params, t):
    """Log likelihood for i.i.d. NBinom measurements, parametrized
    by beta1, beta2."""
    beta_1, beta_2 = params
    dB = beta_2 - beta_1

    if beta_1 <= 0 or beta_2 <= 0:
        return -np.inf
    if not (dB >= 0):
        return -np.inf
    # If beta 1 ~ beta 2, we use a different pdf with just beta_1
    if np.abs(beta_1 - beta_2) < .0001:
        pdf = (beta_1**2) * t * np.exp(-beta_1*t)
    
    else: 
        c = (beta_1 * (beta_1 + dB))/dB
        pdf = c * np.exp(-beta_1*t)*(1-np.exp(-dB*t))

    return np.sum(np.log(pdf))


def mle_iid_exp(times):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by beta1, beta2"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_exp(params, times),
            x0=np.array([.01, .011]),
            args=(times,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)



def gen_fun_exp(params, n, size, rg):
    beta_1, beta_2 = params
    b_1 = rg.exponential(1/beta_1, size)
    b_2 = rg.exponential(1/beta_2, size)
    return b_1 + b_2

def exp_bootstrap(data):
	bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
    mle_iid_exp,
    gen_fun_exp,
    data,
    gen_args=(data, ),
    size=2000,
    n_jobs=3,
    progress_bar=False,
	)
	return bs_reps

def exp_conf(data):
	bs_reps = exp_bootstrap(data)
	return np.percentile(bs_reps, [2.5,97.5], axis = 0)

def exp_dist(data):
	conf = exp_conf(data)
	mle = mle_iid_exp(data)
	print('''beta_1
      MLE: {}
      95% Confidence Interval {}
          '''.format(mle[0], conf[:, 0]))
	print('''beta_2
      MLE: {}
      95% Confidence Interval {}
          '''.format(mle[1], conf[:, 1]))
	return

def plot_ecdf_exp(data):
    
    t = data.loc[data['labeled'], 'time to catastrophe (s)'].values 
    mle = mle_iid_exp(t)
    
    bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
        mle_iid_exp,
        gen_fun_exp,
        t,
        gen_args=(t, ),
        size=1000,
        n_jobs=3,
        progress_bar=True,
    )

    conf = np.percentile(bs_reps, [2.5, 97.5], axis=0)
    
    p = iqplot.ecdf(t, q='t (s)', conf_int=True)

    t_theor = np.linspace(0, 2000, 200)
    cdf = st.gamma.cdf(t_theor, 2, loc=0, scale=1/mle[1])
    p.line(t_theor, cdf, line_width=2, color='orange')

    bokeh.io.show(p)
    
    print('''alpha
        MLE: {}
        95% Confidence Interval {}
          '''.format(mle[0], conf[:, 0]))
    print('''beta
        MLE: {}
        95% Confidence Interval {}
              '''.format(mle[1], conf[:, 1]))












