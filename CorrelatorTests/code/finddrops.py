""" Find drops or jumps in amp like VK found finddrops CBID bchan echan """
import numpy as N
import os, sys, libs, mylibs, glob, katdal, time, pickle
from myrfimask import myrfimask as myrfi
from katsdpcal.calprocs import get_bls_lookup
from scipy import signal as ss

import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()
  
def dofind(fn, bchan, echan, scans):

 udtit = ['up', 'down']; minmax = [N.max, N.min]
 num = fn.split('_')[0]
 print(num, ' ',end='')
 f = katdal.open(fn)
 f.select(scans='track',corrprods='cross', pol='h,v')
 print(f.shape)
 print("BCHAN = ",bchan)
 print("ECHAN = ",echan)

 for iud,ud in enumerate([1,-1]):
  print("Detecting amp jump", udtit[iud])
  f.select(scans='track',corrprods='cross', pol='h,v')
  na = len(f.ants); nbl = int(na*(na-1)/2)
  nchan = f.shape[1]
  freqs = f.freqs
  freq0 = freqs[0]
  bw = (freqs[1]-freq0)*nchan
  size = 15
  file1 = open(num"_"+udtit[iud]+"_baddump_file","w")

  # Get antenna names etc
  ant_names = N.asarray([a.name for a in f.ants])
  cross_blslook = get_bls_lookup(ant_names, f.corr_products)[:nbl]
  index0=[cp[0] for cp in cross_blslook]
  index1=[cp[1] for cp in cross_blslook]
  index0, index1 = N.asarray(index0), N.asarray(index1)
  antnums = N.asarray([int(a[1:]) for a in ant_names])
  flagchans = mylibs.get_rfi_chans(freq0, bw, nchan, drop=0.1)
  
  t1 = time.time()
  if scans==None: scans = f.scan_indices
  n1,m1 = libs.subplot(len(scans)); 
  
  fs = 25
  template = []; allpeaks=[]; allfluxes = []
  thresh = [7., 6., 5.]
  badnums = N.zeros((2,nbl,f.shape[0]+len(scans)))
  badnumsant = N.zeros((2,na,f.shape[0]+len(scans)))
  timectr = 0
  for iscan,scan in enumerate(scans):
    print('  Doing scan', scan, end=': '); sys.stdout.flush()
    peaks, fluxes = [], []
    f1 = pl.figure(figsize=(20,20))
    f2 = pl.figure(figsize=(20,20))
    f.select(scans=scan)
    vis = f.vis[:,bchan:echan,:]  # time chan blXpol
    flags = f.flags[:,bchan:echan,:]
    vis = N.abs(vis)
    vis = vis*N.where(flags,N.nan,1)
    del flags
    vis = N.asarray([vis[:,:,:nbl],vis[:,:,nbl:]])  # pol time chan bl
    for flag in flagchans:
      vis[:,:,flag[0]:flag[1],:] = N.nan
    print('done reading data', end=' ... '); sys.stdout.flush()
  
    for ipol in range(2):
      peak, flux = [], []
      alldata = []; alldiff = []
      pl.figure(f1.number); pl.subplot(2,1,ipol+1); 
      pl.figure(f2.number); pl.subplot(2,1,ipol+1); 
      for ibl in range(nbl):
        flux.append(N.nanmedian(vis[ipol,:,:,ibl]))
        d = N.copy(vis[ipol,:,:,ibl])
        bp = N.nanmean(d,0)
        for itime in range(d.shape[0]): d[itime] -= bp
        d = N.nanmean(d,1)  # time series
        mm,cc = N.polyfit(N.arange(len(d)),d,1); d = d - mm*N.arange(len(d))-cc

        diff = N.concatenate((N.zeros(1),N.asarray([d[i]-d[i-1] for i in range(1,len(d))])))
        alldiff.append(diff)
        alldata.append(d)
      alldiff = N.asarray(alldiff)
      alldata = N.asarray(alldata)
      med = N.nanmedian(alldata)
      mad = 1.5*N.nanmedian(N.abs(alldata-med))
      med1 = N.nanmedian(alldiff); mad1 = 1.5*N.nanmedian(N.abs(med1-alldiff))

      nthr = N.asarray([len(N.where((alldata.flatten()-med)*ud>thresh[kk]*mad)[0]) for kk in range(len(thresh))])
      avsp = N.zeros(vis.shape[1]); nn1 = 0
      thr = 0; takebls = []
      if N.max(nthr)>0:
        thr = thresh[N.where(nthr>0)[0][0]]
        goodinds = []
        for ibl in range(nbl):
          goodind = []
          pl.figure(f2.number); pl.plot(alldata[ibl], lw=0.6)
          if minmax[iud](alldata[ibl]-med)*ud>thr*mad and (minmax[iud](alldiff[ibl]-med1)*ud>5.0*mad1):

           take = False
           inddata = N.where((alldata[ibl]-med)/mad*ud>thr)[0]
           inddiff = N.where((alldiff[ibl]-med1)/mad1*ud>5.0)[0]
           for ind in inddata:
             if ind in inddiff: 
               take = True; goodind.append(ind)
               file1.write("%d %d %f %d %s %s\n" %(scan,ipol,f.timestamps[ind],ibl,ant_names[index0[ibl]],ant_names[index1[ibl]]))
           if take: takebls.append(ibl)
          goodinds.append(goodind)

        if len(takebls)>3:
          for ibl in takebls:
            avsp += alldata[ibl]; nn1 += 1
            xx = N.where((alldata[ibl]-med)/mad*ud>thr,(alldata[ibl]-med)/mad,0)
            inds = ss.find_peaks(xx, distance=5)[0]

            badnums[ipol,ibl,inds+timectr] = 1
            badnumsant[ipol,index0[ibl],inds+timectr] += 1
            badnumsant[ipol,index1[ibl],inds+timectr] += 1
            pl.figure(f1.number)
            pl.subplot(2,1,ipol+1); 
            pl.plot(alldata[ibl], lw=0.6)
            pl.plot(goodinds[ibl],alldata[ibl][goodinds[ibl]],'or',ms=10)

            for iind,ind in enumerate(goodinds[ibl]):
              peak.append(alldata[ibl][ind])
              if iind<len(inds)-1:
                x = alldata[ibl][ind-2:min(ind-2+size,inds[iind+1])]
              else:
                x = alldata[ibl][ind-2:min(ind-2+size,len(alldata[ibl]))]
              xx = N.zeros(size); xx[:len(x)] = x; mx = N.max(xx)
              if mx*ud>0: xx = xx/mx
              template.append(xx)
      peaks.append(peak); fluxes.append(flux)
      print('calculating jumps for pol',ipol,end=' ... '); sys.stdout.flush()
      for ff in [f1,f2]:
        pl.figure(ff.number)
        pl.plot(avsp/nn1, 'k', lw=2)
        pl.xlabel('Dump index', fontsize=fs)
        pl.ylabel('Amp (scaled)', fontsize=fs)
        pl.title('Pol '+str(ipol)+'; thresh='+str(int(thr)), fontsize=fs)
        pl.xticks(fontsize=fs); pl.yticks(fontsize=fs)
    del vis
    allpeaks.append(peaks); allfluxes.append(fluxes)
    pl.figure(f1.number)
    pl.suptitle(num+' '+udtit[iud]+' average bad amp for scan '+str(scan),fontsize=fs)
    pl.savefig(num+'_'+udtit[iud]+'_dump_scan_'+str(scan)+'_bad.png')
    pl.figure(f2.number)
    pl.suptitle(num+' '+udtit[iud]+' average all amp for scan '+str(scan),fontsize=fs)
    pl.savefig(num+'_'+udtit[iud]+'_dump_scan_'+str(scan)+'_all.png')
    badnums[ipol,ibl,timectr+f.shape[0]] = N.nan
    timectr += f.shape[0]+1; 
    print('plotting')

  print("DONE")
  fs = 12
  pl.figure(figsize=(10,5)); cols = ['b', 'r']
  ratio = []
  for i in range(len(allfluxes)):
    for j in range(2):
      if len(allpeaks[i][j])>0:
        pl.errorbar(scans[i], N.nanmedian(allpeaks[i][j])/N.nanmedian(allfluxes[i][j])*100, \
                    N.nanstd(allpeaks[i][j])/N.nanmedian(allfluxes[i][j])*100, c=cols[j])
        for k in range(len(allpeaks[i][j])): ratio.append(N.nanmedian(N.asarray(allpeaks[i][j]))/N.nanmedian(N.asarray(allfluxes[i][j])))
  ratio = N.asarray(ratio)
  pl.xlabel('Scans', fontsize=fs); pl.ylabel('Percentage of jump peak', fontsize=fs)
  pl.title(num+' average percent of jump amp, '+str("%.3f"%(N.nanmedian(ratio)*100)), fontsize=fs)
  pl.savefig('a1.png')
  print("%s %.4f %s"%("Average strength of peaks is",N.nanmedian(ratio)*100,"%"))
  file1.close()
  pickle.dump([badnums, badnumsant, ant_names, N.asarray(template), N.asarray(allpeaks), N.asarray(allfluxes)], \
               open(num+'_'+udtit[iud]+'_baddumps.pickle','wb'),protocol=2)
  print("%s %.5f %.5f %s" %("Occupancy of peaks is",N.nansum(badnums[0])/N.product(badnums[0].shape)*100,\
        N.nansum(badnums[1])/N.product(badnums[1].shape)*100,"% for 2 pols"))

  for i in range(2):
    pl.figure(figsize=(14,10))
    pl.subplot(211); libs.imshow(N.transpose(badnums[i]))
    pl.ylabel('Baselines',fontsize=fs)
    pl.title('Discrepant time series for baseline vs time, occupancy '+str("%.3f"%(N.nansum(badnums[i])/N.product(badnums[i].shape)*100)), fontsize=fs)
    pl.subplot(212); libs.imshow(N.transpose(badnumsant[i]))
    pl.xlabel('Dumps', fontsize=fs); pl.ylabel('Antennas', fontsize=fs)
    pl.title('Discrepant time series for antenna versus time', fontsize=fs)
    pl.suptitle(num+' '+udtit[iud]+' jump dumps for pol '+str(i), fontsize=fs)
    pl.savefig(num+'_'+udtit[iud]+'_jumpdumps_baseline_antenna_vs_time_pol'+str(i)+'.png', pad_inches=0.1, bbox_inches = 'tight')
    
  pl.figure(figsize=(16,6))
  pl.plot(N.sum(badnumsant[0],1))
  pl.plot(N.sum(badnumsant[1],1))
  aname = [ant_names[i][2:] for i in range(na)]
  pl.xticks(N.arange(na), aname, rotation=45)
  pl.xlabel('Ant names', fontsize=fs)
  pl.ylabel('Number of amp jumps', fontsize=fs)
  pl.title(num+' '+udtit[iud]+' number of amp jumps per antenna for pol XX', fontsize=fs)
  pl.savefig(num+'_'+udtit[iud]+'_jumpdumps_num_vs_ant.png', pad_inches=0.1, bbox_inches = 'tight')
  
  pl.figure(figsize=(12,6))
  pl.plot(N.mean(template,0))
  pl.title(num+' '+udtit[iud]+' stacked average of all 8550 discrepant time series cut outs; strength is '+str("%.3f"%(N.nanmedian(ratio)*100)))
  pl.xlabel('Relative dump', fontsize=fs); pl.ylabel("Stacked amplitudes", fontsize=fs)
  pl.savefig(num+'_'+udtit[iud]+'_baddumps_templates_av.png', pad_inches=0.1, bbox_inches = 'tight')
  

if __name__ == '__main__':
  fn = sys.argv[1]
  bchan = int(sys.argv[2])
  echan = int(sys.argv[3])
  if len(sys.argv) > 4:
    scans = [int(x) for x in sys.argv[4:]]
  else: scans = None
  num = fn.split('_')[0]
  dofind(fn, bchan, echan, scans)


