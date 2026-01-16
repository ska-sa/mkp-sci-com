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
Usage: run sdp_bandpass.py rdbfilename gainsolve refant norm
Eg   : run sdp_bandpass.py 150000000_sdp_l0.rdb scan 0 False

gainsolve = dump: solve for every integration
          = scan: solve for every scan
          = int: solve for solint = int integrations
norm = normalise bandpasses or not.
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
gainsolve = sys.argv[2]
refant = int(sys.argv[3])
norm = sys.argv[4]
norm = norm=='True'
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
pols = ['hh','vv']
for k in range(1): # dummy
    bps = []
    f.select(corrprods='cross', pol='hh,vv', scans='track')
    if gainsolve=='dump':
      print("Calculate for dump", end=' ')
      dum = int(pow(10,int(floor(N.log10(f.shape[0])))-1))*2
      for dump in range(f.shape[0]):
        if dump%dum==0: print(dump, end=' '); sys.stdout.flush()
        vis = f.vis[dump,:,:]
        flag = f.flags[dump,:,:]
        flag = N.where(flag, 0, 1)
        vis = vis*flag
        vis = N.asarray([vis[:,:d.nbl],vis[:,d.nbl:]])
        bp = bp_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant).compute()
        if norm:
          norm = normalise_complex(bp) # normalise phase=0 at centre and av amp=1 on axis=0
          bp *= norm
        bps.append(bp)

      print()

    if gainsolve=='scan':
      print("Calculate for scan", end=' ')
      for scan in scans:
        print(scan, end=' '); sys.stdout.flush()
        f.select(scans=[scan])
        vis = f.vis[:]
        flag = f.flags[:]
        flag = N.where(flag, 0, 1)
        vis = N.mean(vis*flag,0)
        vis = N.asarray([vis[:,:d.nbl],vis[:,d.nbl:]])
        bp = bp_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant).compute()
        if norm:
          norm = normalise_complex(bp) # normalise phase=0 at centre and av amp=1 on axis=0
          bp *= norm
        bps.append(bp)
      print()

    if gainsolve=='average':
      print("Calculate for index", end=' ')
      for dump in range(0,f.shape[0],solint):
        print(dump, end=' '); sys.stdout.flush()
        vis = f.vis[dump:min(dump+solint,f.shape[0]),:,:]
        flag = f.flags[dump:min(dump+solint,f.shape[0]),:,:]
        flag = N.where(flag, 0, 1)
        vis = N.mean(vis*flag,0)
        vis = N.asarray([vis[:,:d.nbl],vis[:,d.nbl:]])
        bp = bp_fit(vis, N.ones(vis.shape), cross_blslook, refant=refant).compute()
        if norm:
          norm = normalise_complex(bp) # normalise phase=0 at centre and av amp=1 on axis=0
          bp *= norm
        bps.append(bp)
      print()

t2 = time.time()
bps = N.asarray(bps)
bps = N.transpose(bps,(1,0,3,2))
N.save("bp_sdp_"+gainsolve, bps)

err, snr, flag = N.ones(bps.shape), N.ones(bps.shape), N.ones(bps.shape)
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

print('bps',bps.shape)
names = N.asarray([f.ants[i].name for i in range(nant)])
field = N.zeros(bps.shape[1])
scans = N.zeros(bps.shape[1])
antpos = N.asarray([f.ants[i].position_ecef for i in range(nant)])
N.save('enu', antpos)

output = bps, antpos, names, err, snr, flag, u, v, (index0,index1,antnums,antind), [scans, nant, field, nchan, freq0, bw]
if gainsolve == 'average': gainsolve = 'av'+str(solint)
print("\nSending bandpasses for plotting")
mylibs.plot_bandpass(output, calname = 'bp_sdp_'+gainsolve)


