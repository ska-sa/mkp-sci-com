#l Read output of docasa_64chan_calib.py
import os, sys, glob, pickle, warnings, libs, mylibs
import corr_test_chans as cc
import numpy as N
import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ion()
warnings.filterwarnings("ignore")

opfiles = './'

fns = glob.glob(opfiles+'op_*')
fns = N.asarray([int(fn.split('_')[-1]) for fn in fns])
nfiles = N.max(fns)
print("Num of scans", nfiles)

fns = glob.glob(opfiles+'op_*')
d = pickle.load(open(fns[0], 'rb'), encoding='bytes')[0][0]
dum = len(d)
print("Num of chans", dum)
if dum>4000: nch = 32
else: nch = 4

extns = ['raw', 'self'] 
add = ['bp'+ str(d1+1) for d1 in range(nfiles)]
extns += add
fns = glob.glob("*.rdb")
numid = fns[0].split('_')[0]
print("Processing", numid)

fband = cc.myfullband['l'][nch]
fbandfl = cc.myfullbandflags['l'][32]

xdip = 64-(fband[0]%64)
print("Location of dip is", xdip)
fold = 64

rawrms = []
for j in range(1,nfiles+1):
  fn1 = opfiles+'op_64chan_raw_scan_'+str(j)
  if 'raw' in fn1: 
    d = pickle.load(open(fn1, 'rb'), encoding='bytes')
    rawrms.append(d[1])
rawrms = N.asarray(rawrms)
pl.figure()
pl.plot(rawrms[:,0])
pl.plot(rawrms[:,1])
pl.title(numid+' raw rms per scan')
pl.savefig(numid+'_rawrms_scan.png')

bad1 = bad2 = []
scanl = 10.0
if numid == '1586078244':
  bad1 = []
  bad2 = []
  scanl = 15.0
if numid == '1573662272':
  bad1 = []
  bad2 = [1, 2, 18, 5, 6, 8, 23, 34, 37]
  scanl = 10.0
if numid == '1574617721':
  bad1 = [1, 2, 3, 6, 7, 14, 18, 19, 26, 27, 28, 29, 41, 42, 46, 47, 63, 72]
  bad2 = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,15,16,17,14,20,21, 18, 19, 26, 27, 28, 29, 41, 42, 46, 47, 63, 72]
  scanl = 5.0
if numid == '1589558455':
  bad1 = []
  bad2 = [14,15,16,17,18,19,20,21,23,25,27,29,31,33]
  scanl = 10.
badt = [str(len(bad1)), str(len(bad2))]

for ibad,bad in enumerate([bad1]): #, bad2]):
  allspecs = []; allfiles = []; allsnrs = []; allrmss = []; scannums = []
  for ii,extn in enumerate(extns):
    fns, snrs, specs, files, rmss = [], [], [], [], []
    for j in range(1,nfiles+1):
      fn1 = opfiles+'op_64chan_'+extn+'_scan_'+str(j)
      if os.path.isfile(fn1): 
        fns.append(fn1)
        if ii==0: scannums.append(j)
    for fn in fns:
      d = pickle.load(open(fn, 'rb'), encoding='bytes')
      spec = d[0] 
      snr = N.ones(2)
      for ipol in range(2):
        med = N.nanmedian(spec[ipol]); mad = 1.5*N.nanmedian(N.abs(spec[ipol]-med))
        ind = N.where(N.abs(spec[ipol]-med)/mad>3.0)[0]
        spec[ipol][ind] = N.nan
        #for ifl in range(len(fbandfl)):
        #  spec[ipol][fbandfl[ifl][0]/8-fband[0]:fbandfl[ifl][1]/8-fband[0]] = N.nan
    
        for badd in bad: 
          if ('_bp'+str(badd)+'_' in fn) or (fn.split('_')[-1]==str(badd)): 
            spec[ipol] = spec[ipol]*N.nan
  
        # Get dips
        folded = N.asarray([N.nanmean(spec[ipol,k::fold]) for k in range(fold)])
        num = folded[xdip]
        folded[xdip] = N.nan
        den = N.nanstd(folded)
        snr[ipol] = num/den
      
      take = True
      for badd in bad: 
        if ('_bp'+str(badd)+'_' in fn) or (fn.split('_')[-1]==str(badd)): take = False
      if take:
        rms1 = d[1]/rawrms[int(fn.split('_')[-1])-1]
      else:
        rms1 = [N.nan, N.nan]
      rmss.append(rms1)
      specs.append(spec)
      files.append(fn)
      snrs.append(snr)
    allspecs.append(specs)
    allfiles.append(files)
    allsnrs.append(snrs)
    allrmss.append(rmss)
  allspecs, allfiles = N.asarray(allspecs), N.asarray(allfiles)
  allsnrs, allrmss = N.asarray(allsnrs), N.asarray(allrmss)
  ngscan = len(scannums)
  print(allrmss.shape, allspecs.shape)

  pl.figure()
  for i in range(2): 
    pl.subplot(2,1,i+1);libs.imshow(allrmss[2:,:,i])
    pl.colorbar()
  pl.suptitle('RMS of bp (x) on scan (y), flag '+badt[ibad]+' scans')
  pl.savefig(numid+'_rms_matrix_flag'+badt[ibad]+'.png')
  
  nn,mm = libs.subplot(nfiles)
  pl.figure(figsize=(15,15))
  for i in range(nfiles):                                                       
      pl.subplot(nn,mm,i+1)                                                  
      pl.plot(allspecs[1][i][0], lw=1) 
      pl.plot(allspecs[1][i][1], lw=1)
      pl.axis([0,len(allspecs[1][i][0]),N.nanmin(allspecs[1]),N.nanmax(allspecs[1])])           
      pl.title(str(scannums[i]))                                             
      #pl.xticks([]); pl.yticks([])  
  pl.suptitle(numid+' av spec after selfbp per scan after flag, flag '+badt[ibad]+' scans')
  pl.savefig(numid+'_avspec_selfbp_afterflag_flag'+badt[ibad]+'.png')
  
  pl.figure(figsize=(15,15))
  for i in range(nfiles):                                                       
      pl.subplot(nn,mm,i+1)                                                  
      pl.plot(allspecs[0][i][0], lw=1)
      pl.plot(allspecs[0][i][1], lw=1)
      pl.axis([0,len(allspecs[0][i][0]),N.nanmin(allspecs[0]),N.nanmax(allspecs[0])])           
      pl.title(str(scannums[i]))                                             
      #pl.xticks([]); pl.yticks([])  
  pl.suptitle(numid+' av spec raw per scan after flag, flag '+badt[ibad]+' scans')
  pl.savefig(numid+'_avspec_raw_afterflag_flag'+badt[ibad]+'.png')
  
  title = ['raw', 'self']
  for ipol in range(2):
    pl.figure(figsize=(12,6))
    for i in range(2):
      pl.subplot(2,2,i+1)
      for j in range(nfiles): 
        pl.plot(allspecs[i][j][ipol], lw=1)
      pl.title(title[i])
      pl.plot(N.nanmean(allspecs[i,:,ipol,:],0), 'k-', lw=3)
    for i in range(2):
      pl.subplot(2,2,i+3)
      dum = N.copy(allspecs[i,:,ipol,:])  # scan chan
      for j in range(dum.shape[0]):
          dum[i] = dum[i] - mylibs.poly_filter(dum[i],30)
      dum1= N.nanmean(dum,0)
      dum1 = mylibs.fold(dum1, 64)
      pl.plot(dum1)
      dum1 = mylibs.fold(dum1-mylibs.poly_filter(dum1,30),64)
      #pl.plot(dum1)
      pl.title(title[i])
    pl.suptitle(numid+' av spec raw and self and after fold; pol '+str(ipol)+ ' flag '+badt[ibad]+' scans')
    pl.savefig(numid+'_avspec_rawself_afterflag_fold_pol'+str(ipol)+'_flag'+badt[ibad]+'.png')
  
  fields = []
  fn2 = glob.glob('fields')
  if len(fn2)==1:
    fd = open(fn2[0])
    for line in fd:
      fields.append(N.asarray(line.strip().split(' '),int))

  if len(fields)>0:  
    title = ['raw', 'self']
    pl.figure(figsize=(12,6))
    for ifd in range(len(fields)):
      for ipol in range(2):
        pl.subplot(len(fields),2,ifd*2+1+ipol)

        spec = N.zeros(allspecs.shape[3])
        for ii in range(allspecs.shape[1]):
            if ii+1 in fields[ifd]:
              spec += allspecs[0,ii,ipol]
        
        spec = mylibs.fold(spec-mylibs.poly_filter(spec,100),64)
        pl.plot(spec)
    pl.suptitle(numid)
    pl.savefig(numid+'TRY.png')


  # Plot snrs
  for ipol in range(2):
    pl.figure(figsize=(10,8))
    pl.subplot(211)
    for i in range(len(extns)):
      if i<2:
        pl.plot(allsnrs[i,:,ipol], '.-', label=extns[i])  # bp, scan, pol
      else:
        pl.plot(allsnrs[i,:,ipol], '.-')
    pl.plot([0,nfiles], [0,0], 'k-')
    pl.legend()
    pl.xlabel('Scan number'); pl.ylabel('64-chan dip SNR')

    fn2 = glob.glob('fields')
    pl.subplot(212)
    if len(fn2)==0:
      for i in range(len(extns)):
        if i<2:
          pl.semilogy(allrmss[i,:,ipol], '-.', label=extns[i])
        else:
          pl.semilogy(allrmss[i,:,ipol], '.-')
      pl.legend()

    if len(fn2)==1:
      mk = {}; mk[True] = ['r', 'm', 'orange']; mk[False] = ['b', 'g', 'c']
      fd = open(fn2[0])
      fields = []
      for line in fd:
        fields.append(N.asarray(line.strip().split(' '),int))
      rank = N.zeros((36,34))  
      for i in range(2,len(extns)):
        for j in range(allrmss.shape[1]):
          if True:
            if allrmss[i,j,0] > 0.1: rank[i,j] = 1
            same = False; iifd = 9
            for ifd in range(len(fields)):
              if (j+1 in fields[ifd]) and (i-2+1 in fields[ifd]): same = True
              if j+1 in fields[ifd]: iifd = ifd
            if same:
              pl.semilogy(j, allrmss[i,j,ipol], 'o', color=mk[same][iifd])
              if ipol==0 and allrmss[i,j,ipol]>0.1: print(i, j, ipol, iifd, allrmss[i,j,ipol])
            else:
              pl.semilogy(j, allrmss[i,j,ipol], 'o', color=mk[same][iifd])
      for i in range(2):
          pl.semilogy(allrmss[i,:,ipol])
      N.save('rank', rank)

    pl.xlabel('Scan number'); pl.ylabel('RMS')
    pl.suptitle(numid+' 32k 4dec 64chan dip SNR and RMS; pol '+str(ipol)+ ' flag '+badt[ibad]+' scans')
    pl.savefig(numid+'_64chan_dip_SNR_RMS_end_pol'+str(ipol)+'_flag'+badt[ibad]+'.png')
  
  dum1 = N.ones((2,len(extns), nfiles))*N.nan
  values = [[],[]]
  for ipol in range(2):
    avrms = N.ones(nfiles)
    stdrms = N.ones(nfiles)
    for i in range(nfiles): 
      values[ipol].append([])
    for i in range(2,len(extns)):
      x = allrmss[i,:,ipol]
      self = int(extns[i][2:])
      for j in range(ngscan):
        dist = abs(scannums[j]-self)
        values[ipol][dist].append(x[j])
        dum1[ipol,i,j] = x[j]
  
  pl.figure(figsize=(12,12))
  for ipol in range(2):
    for i in range(nfiles):
      avrms[i] = N.nanmean(N.asarray(values[ipol][i]))
      stdrms[i] = N.nanstd(N.asarray(values[ipol][i]))
    x = N.arange(len(avrms))*scanl/60
    pl.subplot(2,2,ipol*2+1)
    pl.errorbar(x, avrms, stdrms, c='r')
    pl.xlabel('Diff in hrs'); pl.ylabel('RMS'); pl.title('After applying BP')
    pl.subplot(2,2,ipol*2+2)
    #x1 = N.searchsorted(x, 0.2)
    #x2 = N.searchsorted(x, 2.5)
    #pl.errorbar(x[x1:x2], avrms[x1:x2], stdrms[x1:x2], c='r')
    for i in range(nfiles):
      pl.plot([i*scanl/60]*len(values[ipol][i]), values[ipol][i], '.')
    pl.plot(x, avrms, 'r', lw=4)
    pl.xlabel('Diff in hrs'); pl.ylabel('RMS'); pl.title('With all data')
    pl.suptitle(numid+' 32k 4dec; rms of scans after applying BP hrs away, flag '+badt[ibad]+' scans')
    pl.savefig(numid+'_rms_scans_bp_inhrs_end_flag'+badt[ibad]+'_flag'+badt[ibad]+'.png')
     
  
  
# Plot all specs to see how bad they are
nn,mm = libs.subplot(nfiles)
for i in range(len(extns)):
  pl.figure(figsize=(8,8))
  mn,mx = N.nanmin(allspecs[i]), N.nanmax(allspecs[i])
  for j in range(nfiles):
    pl.subplot(nn,mm,j+1)
    pl.plot(allspecs[i,j])
    pl.axis([-50,7550,mn,mx])
    pl.title(allfiles[i][j].split('_')[-1])
  pl.suptitle(numid+' 32k 4dec 64chan '+allfiles[i][0].split('_')[2])
  #pl.savefig(numid+'_64chan_allspecs_'+extns[i]+'.png')

# Plot folded specs for each, fold = 128
for i in range(len(extns)):
  pl.figure(figsize=(8,8))
  for j in range(nfiles):
    pl.subplot(nn,mm,j+1)
    folded = N.asarray([N.nanmean(allspecs[i,j][k::fold]) for k in range(fold)])
    pl.plot(folded)
    pl.plot(xdip,folded[xdip], 'o')
    pl.title(allfiles[i][j].split('_')[-1])
  pl.suptitle(numid+' 32k 4dec folded '+allfiles[i][0].split('_')[2])
  #pl.savefig(numid+'_64chan_allfolded_'+extns[i]+'_end.png')

  
  
