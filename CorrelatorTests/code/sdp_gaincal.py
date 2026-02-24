import numpy as N
from math import *
import os, pickle, warnings, libs, mylibs, time, sys, pprocess, glob
from scipy.ndimage.filters import median_filter

import katdal
import dask.array as da
import casacore.tables as t
from katsdpcal.calprocs import get_bls_lookup, k_fit, ants_from_bllist, normalise_complex, g_fit
from katsdpcal.calprocs_dask import wavg_full, bp_fit, wavg_full_t
warnings.filterwarnings("ignore")

"""
Calculates gains using g_fit of sdp 
Usage: run sdp_gaincal.py rdbfilename bchan numchan gainsolve refant
Eg   : run sdp_gaincal.py 150000000_sdp_l0.rdb 18000 64 dump 0

gainsolve = dump: solve for every integration
          = scan: solve for every scan
          = int: solve for solint = int integrations
"""

basedir = '/data/mohan/'
basedir += '/'

import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()


class myclass(object):
    pass
d = myclass()

fns = sys.argv[1]  # rdb file name
ichan=int(sys.argv[2]) # bchan
dchan = int(sys.argv[3]) # num of chans (echan-bchan+1)
gainsolve = sys.argv[4]
refant = sys.argv[5]
if gainsolve not in ['dump', 'scan']:
  try: solint = int(gainsolve)
  except: raise RuntimeError("gainsolve can be 'dump'/'scan'/integer")
  gainsolve = 'average'

fname = fns.split('_')[0]
d.doc = False
mylibs.create_dirs(d, fname, basedir)
os.chdir(basedir+'/'+fname)

casa = False
def my_apply(sol, vis, bls_lookup):
    inv_solval=N.reciprocal(sol)

    index0=[cp[0] for cp in bls_lookup]
    index1=[cp[1] for cp in bls_lookup]
    correction=inv_solval[...,index0] * inv_solval[...,index1].conj()
    
    return correction * vis

t1 = time.time()
f = katdal.open(d.fdir+fns)
t2 = time.time()
print("Done opening", d.fdir+fns," in ", t2-t1, "secs")
scans = f.scan_indices

f.select(corrprods='cross', pol='hh,vv', scans='track')
mylibs.read_katdal_para(f, d)

f.select()
f.select(scans='track')
ant_names = [a.name for a in f.ants]
cross_blslook = get_bls_lookup(ant_names, f.corr_products)[d.na*4:d.na*4+d.nbl]
index0=[cp[0] for cp in cross_blslook]
index1=[cp[1] for cp in cross_blslook]
index0, index1 = N.asarray(index0), N.asarray(index1)
antnums = [int(a[1:]) for a in ant_names]

t1 = time.time()
f.select(corrprods='cross', pol='hh,vv', scans='track')
print("Shape of file", f.shape)
all_g = []
pols = ['hh','vv']
for j in range(2):
    gs = []
    f.select(corrprods='cross', pol=pols[j], scans='track')
    if gainsolve=='dump':
      print("Calculate for pol",j,"dump", end=' ')
      dum = int(pow(10,int(floor(N.log10(f.shape[0])))-1))*2
      for dump in range(f.shape[0]):
        if dump%dum==0: print(dump, end=' ')
        vis = f.vis[dump,ichan:ichan+dchan,:]
        flag = f.flags[dump,ichan:ichan+dchan,:]
        flag = N.where(flag, 0, 1)
        vis = N.mean(vis*flag,0)
        gsol = g_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant)
        gs.append(gsol)
      print()

    if gainsolve=='scan':
      print("Calculate for pol",j,"scan", end=' ')
      for scan in scans:
        print(scan, end=' ')
        f.select(scans=[scan])
        vis = f.vis[:,ichan:ichan+dchan,:]
        flag = f.flags[:,ichan:ichan+dchan,:]
        flag = N.where(flag, 0, 1)
        vis = N.mean(N.mean(vis*flag,1),0)
        gsol = g_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant)
        gs.append(gsol)
      print()

    if gainsolve=='average':
      print("Calculate for pol",j,"index", end=' ')
      for dump in range(0,f.shape[0],solint):
        print(dump, end=' ')
        vis = f.vis[dump:dump+solint,ichan:ichan+dchan,:]
        flag = f.flags[dump:dump+solint,ichan:ichan+dchan,:]
        flag = N.where(flag, 0, 1)
        vis = N.mean(N.mean(vis*flag,1),0)
        gsol = g_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant)
        gs.append(gsol)
      print()
    all_g.append(gs)


t2 = time.time()
all_g = N.asarray(all_g)
N.save("sn_sdp_"+gainsolve, all_g)

err, snr, flag = N.ones(all_g.shape), N.ones(all_g.shape), N.ones(all_g.shape)
f.select()
f.select(scans='track',corrprods='cross', pol='h')
u, v = f.u, f.v
u = u[int(u.shape[0]/2),:]
v = v[int(v.shape[0]/2),:]
nchan = len(f.channels)
freq0 = f.freqs[0]
bw = f.freqs[-1]-f.freqs[0]

nant = len(f.ants); nbl = int(nant*(nant-1)/2)
antind = N.asarray(N.ones((nant,nant))*999,int)
for i in range(nbl):
  antind[index0[i],index1[i]] = i
  antind[index1[i],index0[i]] = i

names = N.asarray([f.ants[i].name for i in range(nant)])
field = N.zeros(all_g.shape[1])
scans = N.zeros(all_g.shape[1])
antpos = N.asarray([f.ants[i].position_ecef for i in range(nant)])
N.save('enu', antpos)

output = all_g, antpos, names, err, snr, flag, u, v, (index0,index1,antnums,antind), [scans, nant, field, nchan, freq0, bw]
if gainsolve == 'average': gainsolve = 'av'+str(solint)
mylibs.plot_gaincal(output, calname = 'sn_sdp_'+gainsolve)


