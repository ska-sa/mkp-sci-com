
import numpy as N
from importlib import reload
from math import *
import os, pickle, warnings, libs, mylibs, time, sys, glob
from scipy.ndimage.filters import median_filter
import corr_test_chans as cc
import corrtests_config as cfg

import katdal
import dask.array as da
import casacore.tables as t
from katsdpcal.calprocs import get_bls_lookup, k_fit, ants_from_bllist, normalise_complex, g_fit
from katsdpcal.calprocs_dask import wavg_full, bp_fit, wavg_full_t
import corrtestslib as ctl

from docx import Document
from docx.shared import Inches

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()

"""
Usage: run corrtests.py CBID write/nowrite
Eg     run corrtests.py 1567656242 write
"""

t1 = time.time()

#Inputs, to be taken out as command line or config file later
basedir = cfg.basedir
basedir += '/'
doc = cfg.doc

# Get CBID from input
fname = ctl.get_cbid(sys.argv)  # is CBID as str
os.chdir(basedir+fname+'/')

##################################################################################
# READ RDB FILE AND WRITE EACH SCAN TO DISK. ALL OTHERS NEED THIS ################
if sys.argv[2] not in ['write','nowrite']:
  raise RuntimeError("Usage: run corrtests.py CBID write/nowrite")
if sys.argv[2]=='write':
  mylibs.pickle_new_data(num, nsave=True, getflags=True, dotimeavg=True, dofullauto=True, \
  dofullcross=True, onlyauto=False, verbose=True, extn=None, basedir='/data/mohan/', bchan=0, echan=0)

##################################################################################
#SET UP STUFF

# Set up object d
d = ctl.setup_objd(fname, basedir, doc)

# Open via katdal, read parameters, put on to d, read scans, define scan params
f = ctl.setup_file(d)

# Set up the sub bands (see corr_test_chans.py)
ctl.setup_bands(d)
##################################################################################

# The setup scripts above need to be run for the scripts below to work

# Plot summary of SDP flagging fraction; read flag files per scan
ctl.sdpflag_summary(d)

# Plot mean and rms of vis and auto amps per scan; read scanav files per scan
ctl.statsbyscan(d)

# Test for 2 channel ringing in amps; read scanav files per scan
# default value for thresh is d.ringthresh = 0.67
ctl.ringing_2chan(d) # , thresh

# Test for 2 dump ringing in dc; read dc files per scan
# default value for thresh is d.ringthresh = 0.67 and mindump=20
ctl.ringing_2dumpdc(d) # , thresh, mindump

# Test for spectral periodicities using fft in auto and cross; read scanav files per scan
# default win=51, thresh=10.0
ctl.spectralperiods(d) # , win, thresh

# Detrend each scanav spectrum for fullband; read scanav files per scan
# default window for detrending is 31 (can be a bit higher, esp for 4K, 32K)
ctl.detrend(d) # , win

# Check for 2 channel ringing in global averaged detrended spectrum
# Needs ctl.detrend to have been run first
ctl.ringing_2chan_single(d)

# Test for corr coeff between X and Y pol for each spec; take detrended scanav spec
# Needs ctl.detrend to have been run first
ctl.xycorrcoeff_spec(d)

# Test for 64 channel periodic dips; take detrended scanav spec
# Needs ctl.detrend to have been run first
ctl.check_64fold(d)

# Do comb analysis for all start,fold combinations; take detrended scanav spec
# Needs ctl.detrend to have been run first
# default mnn=2, nsearch=1, minfold=5, thresh=5.0
ctl.do_comb(d)  #, mnn, nsearch, minfold, thresh

# Do the 64 chan search using Thomas's method
ctl.thomas_64(d)

d.docfull.save(d.docfullname)
d.docsumm.save(d.docsummname)

t2 = time.time()
print("Written out",d.docsummname,'and',d.docfullname)
print("%s %.1f %s\n" %("DONE CORRELATOR TEST IN", t2-t1,"SECS"))
