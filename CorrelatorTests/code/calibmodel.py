import katdal
import pylab as pl
import numpy as N
import katpoint
from numba import jit, guvectorize
from katsdpcal.calprocs import get_bls_lookup, FluxDensityModel, K_ant, add_model_vis
from katsdpcal.calprocs_dask import wavg_full, bp_fit
from katsdpcal import pipelineprocs
import time
import dask.array as da
import glob, libs, sys
import katsdpcal

"""
Usage: run calibmode.py rdbfilename src band bchan echan modeldir
Eg     run calibmodel.py 1500000_sdp_l0.rdb 1939/0408 L/U 18000 18010 /home/nramanujam/python3/

Use plotbl to make plots of sets of baselines by index
"""

fns = sys.argv[1]
f=katdal.open(fns)

#f.select(scans=1, corrprods='cross', pol='h')
f.select(corrprods='cross', pol='h,v')

src = sys.argv[2]
band = sys.argv[3]
bchan, echan = int(sys.argv[4]), int(sys.argv[5])
modeldir = sys.argv[6]
print("File =",fns)
print("Source =",src)
print("Band =",band)
print("Bchan =",bchan)
print("Echan =",echan)
print("Modeldir =",modeldir)

fluxes, targets = [], []
print('Using J'+src+'_'+band+'_fl and J'+src+'_'+band+'_posn')
print('*'*65)
print('***NOTE: Script does not check if the data is indeed for', src,'***')
print('*'*65)
f1 = open(modeldir+'/J'+src+'_'+band+'_fl')
f2 = open(modeldir+'/J'+src+'_'+band+'_posn')
for line in f1:
  line = line.strip()
  fluxes.append(katsdpcal.calprocs.FluxDensityModel(line))
f1.close()
for line in f2:
  line = line.strip()
  targets.append(katpoint.Target(line))
f2.close()

antennas = f.ants
antenna_names = [a.name for a in antennas]
array_position = katpoint.Antenna('', *antennas[0].ref_position_wgs84)
channel_freqs = f.freqs
timestamps = f.timestamps
bls_lookup = get_bls_lookup(antenna_names, f.corr_products)

if src=='0408':
  target = katpoint.Target('J0408-6545, radec target, 4:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)')
if src=='1939':
  target = katpoint.Target('J1939-6342, radec target, 19:39:25.03, -63:42:45.6, (800.0 8400.0 -3.708 3.807 -0.7202)')

uvw = target.uvw(antennas, timestamps, array_position)
uvw = N.array(uvw, N.float32)

# set up model visibility    
ntimes, nchans, nbls = f.vis.shape
nants = len(f.ants)

# currently model is the same for both polarisations
# TODO: include polarisation in models
k_ant = N.zeros((ntimes, echan-bchan, nants), N.complex64)
complexmodel = N.zeros((ntimes, echan-bchan, nbls), N.complex64)

print("Calculate model visibilities (total="+str(len(targets))+'): ', end=' ')
wl = katpoint.lightspeed / f.channel_freqs[bchan:echan]
for i in range(len(targets)):
    if i%int(len(targets)/20)==0: print(i, end=' '); sys.stdout.flush()
    model_targets = targets[i]
    model_fluxes = fluxes[i]
    S=model_fluxes.flux_density(f.channel_freqs[bchan:echan]/1e6)
    
    l, m = target.sphere_to_plane(*model_targets.radec(), projection_type='SIN', coord_system='radec')
    k_ant = K_ant(uvw, l, m, wl, k_ant)
    complexmodel = add_model_vis(k_ant, bls_lookup[:, 0], bls_lookup[:, 1], S.astype(N.float32), complexmodel)

antsepmap = {(ant1.name, ant2.name): N.linalg.norm(ant1.baseline_toward(ant2)) for ant1 in f.ants for ant2 in f.ants}
antsep = N.array([antsepmap[(inp1[:-1], inp2[:-1])] for inp1, inp2 in f.corr_products])
seporder = antsep.argsort()

print('Reading vis')
vis = f.vis[:,bchan:echan,:]
print('Done')
print('\nCreate plots using plotbl(bls,k), where bls is a list of baseline indices and k is the plot number')

pl.ion()

titles = ['Amp', 'Phase']
def plotbl(bls,k):
 for jj in range(2):
  pl.figure(figsize=(16,16))
  n1,m1 = libs.subplot(len(bls))
  for ii,bl in enumerate(bls):
    pl.subplot(n1,m1,ii+1)
    if jj==0:
      s1 = N.mean(N.abs(complexmodel[:,:,bl]),1)
      s2 = N.mean(N.abs(vis[:,:,bl]), 1)
    if jj==1:
      s1 = N.mean(N.angle(complexmodel[:,:,bl],1),1)
      s2 = N.mean(N.angle(vis[:,:,bl], 1),1)
    mm,cc = N.polyfit(N.arange(len(s1)),s1,1); s1 = s1 - mm*N.arange(len(s1))-cc
    mm,cc = N.polyfit(N.arange(len(s2)),s2,1); s2 = s2 - mm*N.arange(len(s2))-cc
    pl.plot(s2); pl.plot(s1); pl.title(str(bl)+': '+str(bls_lookup[bl][0]+1)+'-'+str(bls_lookup[bl][1]+1))
    if ii+2<m1*(n1-1): pl.xticks([])
  pl.suptitle(fns.split('_')[0]+' '+titles[jj]+' '+src+' '+band+'; blue=data; orange=model')
  pl.savefig(fns.split('_')[0]+'_visph_time_fieldmodel_'+titles[jj]+'_'+str(k)+'.png')



