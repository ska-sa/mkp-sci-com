# Use radon transform to detect multiple dips in autocorr as a function of channel, which may occur
# for a few dumps. This is for the 4-chan dips in auto that are sometimes stationary and sometimes
# drift in time. Doing this agnostic to the regularity
#
# run delaytracks_detect 1564042678 200

import numpy as N
from math import *
from importlib import reload
import os, pickle, warnings, libs, mylibs, time, sys, pprocess, glob
from scipy.ndimage.filters import median_filter
import corr_test_chans as cc

import katdal
import dask.array as da
import casacore.tables as t
from katsdpcal.calprocs import get_bls_lookup, k_fit, ants_from_bllist, normalise_complex, g_fit
from katsdpcal.calprocs_dask import wavg_full, bp_fit, wavg_full_t

warnings.filterwarnings("ignore")

basedir = '/data/mohan/'
basedir += '/'

import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()

class myclass(object):
    pass
d = myclass()

reload(mylibs)
#fname = '1564042678'; maxdm=200  # new test with Marcel
#fname = '1562075212'; maxdm=300  # old marcel data to test on
#fname = '1562920079'; maxdm=800  # marcel challenge
#fname = '1565055966'; maxdm = 200  # SMC C07 01
fname = sys.argv[1]
mylibs.create_dirs(d, fname, basedir)

fname = d.fname
pldir = d.pldir
print(d.fdir, d.sdpdir,  d.pldir)

docalib=True; bx=1;ex=None;flatten=True
boxsize=50; thresh1=3.0; threshneg=-4.0; maxdm=100; minsnr=20; axis=1; niter=3;
doabs=False; robust=False; peaksnr=10.0; sidesnr=3.0; dmdiff=3; doflag=True; doplot=False
from delaytracks_config import *

t1 = time.time()
f = katdal.open(d.fdir+d.fname+'_sdp_l0.full.rdb')
t2 = time.time()
print("Done opening"; d.fdir+d.fname, " in ", t2-t1, " secs")

reload(mylibs)
mylibs.read_katdal_para(f, d)

bands = cc.mybands['l'][d.nch]
bandflags = cc.mybandflags['l'][d.nch]
nband = cc.mynband['l'][d.nch]
fullband = cc.myfullband['l'][d.nch]
fullbandflags = cc.myfullbandflags['l'][d.nch]

print(nband, bands, bandflags)

scans = []
f.select(corrprods='cross', pol='hh,vv', scans='track')
t1 = time.time()
for i, s in enumerate(f.scans()):  # d.scans selects that scan data for vis etc as well!!
    scan, state, target = s
    scans.append(scan)
    
d.gscans = scans

d.npol = npol = 2
print(d.gscans, bands)

reload(mylibs)
# Num of scan groups to do 'dedisp'
wdir = d.sdpdir+'/autoants/'
ffs = glob.glob(wdir+'auto_group*npy')
ffs = [(ff.split('/')[-1]).split('_')[1] for ff in ffs]
ngroup = len(N.unique(ffs))
print("Number of groups of scans", ngroup)

reload(mylibs)
def calcdisp(ff, maxdm):
    arr = N.load(ff)
    stores = []
    for iband in range(nband):
        for ipol in range(npol):
            print('pol ',ipol, ff.split('_')[-1].split('.')[0], bands[iband])
            store = mylibs.dedisp(arr[ipol], docalib=docalib, bx=bx,ex=ex, by=bands[iband][0],ey=bands[iband][1], flagy=bandflags[iband], \
                           doplot=doplot, doabs=doabs, doflag=doflag, robust=robust, flatten=flatten, boxsize=boxsize, minsnr=minsnr, thresh1=thresh1,\ 
                           threshneg=threshneg, maxdm=maxdm, niter=niter, peaksnr=peaksnr, sidesnr=sidesnr, dmdiff=dmdiff)
            stores.append(store)
    return ff, stores

reload(mylibs)

allpeaks = [[],[]]
for ig in range(ngroup):
    print("Processing for group ", ig)
    ffs = glob.glob(wdir+'auto_group'+str(ig)+'_ant*m0*npy')
    results = pprocess.Map(limit=64)
    calc = results.manage(pprocess.MakeParallel(calcdisp))
    for ff in ffs:
        calc(ff,maxdm)
    #    r = calcdisp(ff, maxdm)
    #    allpeaks[ig].append(r)
    
    for i,r in enumerate(results):
        allpeaks[ig].append(r)
        
print("\nDone"      )

# allpeaks is [grp1, grp2]
# grp1 = [nants]
# Each of these are (filename, 
#            ([(sizeb0p0,arrb0p0), (sizeb1p1, arrb0p1), (sizeb1p0,arrb0p1),(sizeb1p1,arrb1p1)])

for ig in range(ngroup):
    peaks = allpeaks[ig]
    nant = len(peaks)
    for iband in range(nband):
        for ipol in range(npol):
            alldata = []
            
            for iant in range(nant):
                name = peaks[iant][0]
                peak = peaks[iant][1][iband*npol+ipol]
                antid = int(name.split('_')[-1].split('.')[0][1:])
                
                npeaks = len(peak[1])
                size = peak[0]
                antdata = []
                for ipeak in range(npeaks):
                    antdata.append(peak[1][ipeak])
                if npeaks>0:
                    alldata.append((antid,antdata))
    
            np = len(alldata)  # ants that have data
            if np>0:
                nn, mm = libs.subplot(np)
                if np==1: xs = 6
                else: xs = 10
                pl.figure(figsize=(xs,3*nn))
                for i in range(np):
                    pl.subplot(nn,mm,i+1)
                    data1 = N.asarray(alldata[i][1])
                    fac = maxdm*f.dump_period*1.0/((bands[iband][1]-bands[iband][0])*f.channel_width/1e6)
                    pl.scatter(data1[:,0],data1[:,1]/fac,s=data1[:,2])
                    pl.title('Ant '+str(alldata[i][0]))
                    ylim = (max(N.max(N.abs(data1[:,1])), maxdm/8.0)+5.)/fac
                    pl.axis([0,size[0],-ylim,ylim])
                    pl.plot([0,size[0]],[0,0],'k-')
                    if (i+1)%mm==1: pl.ylabel('sec/MHz')
                    if (i+1)>nn*(mm-1): pl.xlabel('Time')
                    else: pl.xticks([])
                pl.suptitle(d.fname+' Group '+str(ig)+' Band '+str(iband)+' Pol '+str(ipol))
                pl.savefig(d.pldir+d.fname+'_delaydipslopes_group'+str(ig)+'_band'+str(iband)+'_pol'+str(ipol)+'.png')

