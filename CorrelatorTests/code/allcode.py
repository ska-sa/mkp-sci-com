
import numpy as N
from importlib import reload
from math import *
import os, pickle, warnings, libs, mylibs, time, sys, glob
from scipy.ndimage.filters import median_filter
import corr_test_chans as cc

import katdal
import dask.array as da
import casacore.tables as t
from katsdpcal.calprocs import get_bls_lookup, k_fit, ants_from_bllist, normalise_complex, g_fit
from katsdpcal.calprocs_dask import wavg_full, bp_fit, wavg_full_t
import alltestslibs as acl

from docx import Document
from docx.shared import Inches

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()


t1 = time.time()

#Inputs, to be taken out as command line or config file later
basedir = '/data/mohan/'
basedir += '/'


#================================================================
#########  READ AND PLOT CASA GAIN SOLUTIONS ####################

# Read calibration tables of CASA
d = mylibs.read_gains(caltable)

# Read gaincal table of CASA and make multiple diagnostic plots
mylibs.plot_gaincal(caltable, calname=None, num=None, fieldsep=False, t1=0, t2=0)

# Read bandpass table of CASA and make multiple diagnostic plots
mylibs.plot_bandpass(caltable, num=None, doplot=True)

# Read bandpass table and send the chan-averaged data to be plotted as gaincal solns
mylibs.plot_gain_bandpass(caltable, chan, num=None, doplot=True)

# Read bandpass tables caltable1 and caltable2, plot gains from chan1 averaged and chan2 averaged solns resp
mylibs.plot_twogains_bandpass(caltable1, caltable2, chan1, chan2, title)

# Read gaincal table, fft phases and plot the phase of the detected rippled across antennas
mylibs.gainphasephase(caltable)
#================================================================



#================================================================
#########  COMB FUNCTION ANALYSIS SCRIPTS #######################

# Do comb analysis for 1d array x with fold values from nmin to nmax
comb_analysis(x, nmin=2, nmax=200, doerr=None, norm=True)

# Do comb analysis by zooming in to locate non-integer period with zoom factor zooms. Is slow
comb_analysis_zoom(x, nmin=2, nmax=100, minsnr=15, zooms=[20,100])
#================================================================




