# modify docasa_64chan_calib.py for pprocess
# doesnt work, casa doesnt like mp

from __future__ import print_function
import os, sys, glob, mylibs, warnings, pickle, time
import multiprocessing as mp
import numpy as N
import corr_test_chans as cc
from docasa_64chan_config import *
import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()
warnings.filterwarnings("ignore")

print("This script will do bandpass calibration for each scan, apply each bp to all scans, and calculate rms")
print("Assumes that the dataset is 32K. If not, need to modify flagging routines")

print("\nUsing", fname)
print("Scans", bscan,'to', escan)
print("Window for detrending is", win)
print("Beginning chan is", bchan)
print("Gaincal using chans",gbchan,'to',gechan)

bscan = bscan - 1
print("Will gaincal, split and bandpasscal")
print("Will process each scan")
print('Will process scans',bscan+1,'to',escan)
visfile = fname

ms.open(visfile)
md = ms.metadata()
nchan = md.nchan(0)
ms.close()

msmd.open(visfile)
na = msmd.nantennas()
nscans = msmd.nscans()
t = msmd.timesforscans(1,-1,-1)
dump = dump = N.mean((t-N.roll(t,1))[1:])
dump = round(dump*100)/100.
msmd.close()

print("Number of channels", nchan)
print("Number of antennas", na)
print("Number of scans is", nscans)
print()

def getfoldspec(msfile, col):

  os.system("rm -fr del1.ms")
  # get scan duration for av
  ms.open(msfile)
  t = ms.getdata('time', ifraxis=True)['time']
  nt = str(dump)*len(t)
  # now split it out after averaging. much faster than getting whole thing
  default('split')
  vis = msfile
  outputvis = 'del1.ms'
  datacolumn = col
  timebin = str(nt)+'s'
  split(vis=vis,outputvis=outputvis,datacolumn=datacolumn,timebin=timebin)

  ms.open(outputvis)
  data = N.abs(ms.getdata('data')['data']) # pol chan bl        not # time
  ms.close()
  os.system("rm -fr del1.ms")

  rms_spec = N.nanstd(N.nanmean(N.abs(data),2),1) # get two nums, for each pol, without polyfilt

  for ff in cc.myfullbandflags['l'][32]:
    data[:,ff[0]-bchan:ff[1]-bchan,:] = N.nan

  for i in range(data.shape[0]): # pol
    for j in range(data.shape[2]): # bl
      data[i,:,j] = data[i,:,j] - mylibs.poly_filter(data[i,:,j], win)
  dum1 = N.nanmean(data,2)   # pol X chan
  amp_av_filt = N.zeros(dum1.shape)
  for ipol in range(2):
    amp_av_filt[ipol] = dum1[ipol] - mylibs.poly_filter(dum1[ipol], win)
  
  return amp_av_filt, rms_spec
  

def foldbandpass(bscan,nscans):
  # fold all bandpasses
  fold = 64
  for iscan in range(bscan,nscans):
    tb.open('bandpass_scan_'+str(iscan+1))
    d = N.abs(tb.getcol('CPARAM'))
    for ff in cc.mybandflags['l'][32][1]:
      d[:,ff[0]-bchan:ff[1]-bchan,:] = N.nan
    for i in range(d.shape[0]):
      for j in range(d.shape[2]):
        d[i,:,j] -= mylibs.poly_filter(d[i,:,j],win)
    folded = N.zeros((d.shape[0],fold,d.shape[2]))
    for i in range(2):
      for j in range(na):
        folded[i,:,j] = N.asarray([N.nanmean(d[i,ii::fold,:]) for ii in range(fold)])
    folded = N.nanmean(N.nanmean(folded,2),0)
    full = N.nanmean(N.nanmean(d,2),0)
    N.save(str(num)+"_foldedbp_scan_"+str(iscan+1), folded)
    N.save(str(num)+"_fullavbp_scan_"+str(iscan+1), full)
    
def initialcal():
  # GAINCAL ON FULL FILE
  # doing solint=dump because of phase jumps and can then time average in split and save time
  default('gaincal')
  print('Calculating gaincal table')
  vis = visfile
  spw = '0:'+str(gbchan)+'~'+str(gechan)
  solint = str(dump)+'s'
  combine = ''
  gaintype = 'G'
  caltable = 'sn_table_int'
  gaincal(vis=vis,caltable=caltable,solint=solint,combine=combine,gaintype=gaintype,spw=spw)
  
  
  # APPLY SN TABLE TO FULL FILE
  default('applycal')
  print('Applying gaincal table')
  vis = visfile
  gaintable = [caltable]
  flagbackup = False
  applycal(vis=vis,gaintable=gaintable, flagbackup=flagbackup)
  
  # SPLIT EACH SCAN FILE AFTER APPLYING SN TABLE
  print("Splitting out", end=' ')
  for iscan in range(nscans):
    default('split')
    vis = visfile
    datacolumn = 'corrected'
    outputvis = 'split_scan_'+str(iscan+1)+'.ms'
    scan = str(iscan+1)
    print(iscan+1, end=' '); sys.stdout.flush()
    split(vis=vis,outputvis=outputvis,datacolumn=datacolumn,scan=scan)
  print()

      
def calcbp():
  # RUN BANDPASS FOR EACH SCAN
  print("Calculating bandpass for", end=' ')
  for iscan in range(nscans):
    default('bandpass')
    vis = 'split_scan_'+str(iscan+1)+'.ms'
    caltable = 'bandpass_scan_'+str(iscan+1)
    combine = ''
    refant = 'm000'
    solnorm = False
    print(iscan+1, end=' '); sys.stdout.flush()
    bandpass(vis=vis,refant=refant,caltable=caltable,combine=combine,solnorm=solnorm)
  print()

def processscans(bscan, escan):

  for iscan in range(bscan,escan):
    t1 = time.time()
    print("Starting processing scan", iscan+1)
    # Apply bandpass to each
    default('applycal')
    vis = 'split_scan_'+str(iscan+1)+'.ms'
    gaintable = 'bandpass_scan_'+str(iscan+1)
    print("  Apply selfbp", end=','); sys.stdout.flush()
    applycal(vis=vis,gaintable=gaintable,flagbackup=False)
  
    # Get dip and avspec for normal data first
    vis = 'split_scan_'+str(iscan+1)+'.ms'
    print(" getspec raw", end=','); sys.stdout.flush()
    ret = getfoldspec(vis, 'data')
    pickle.dump(ret, open("op_64chan_raw_scan_"+str(iscan+1),"w"))
  
    # Get dip and avspec for data bped by itself
    vis = 'split_scan_'+str(iscan+1)+'.ms'
    print(" getspec self", end=''); sys.stdout.flush()
    ret = getfoldspec(vis, 'corrected')
    pickle.dump(ret, open("op_64chan_self_scan_"+str(iscan+1),"w"))
    print()

    t3 = time.time()
    # apply all bps to all scans
    print("  Apply bandpass and getspec for bp", end=" ")
    for bpscan in range(nscans):
      default('applycal')
      vis = 'split_scan_'+str(iscan+1)+'.ms'
      print(str(bpscan+1), end=','); sys.stdout.flush()
      gaintable = 'bandpass_scan_'+str(bpscan+1)
      applycal(vis=vis,gaintable=gaintable,flagbackup=False)
      # Get dip and avspec for data bped by 1st scan
      vis = 'split_scan_'+str(iscan+1)+'.ms'
      ret = getfoldspec(vis, 'corrected')
      pickle.dump(ret, open("op_64chan_bp"+str(bpscan+1)+"_scan_"+str(iscan+1),"w"))
    print()

    t2 = time.time()
    print("%s %d %s %.1f %s" %("Time for scan", iscan+1, "is",(t2-t1)/60., "min\n"))


initialcal()
calcbp()
processscans(bscan, escan)



