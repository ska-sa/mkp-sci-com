# Do gaincal and bandpass in CASA
import glob, libs, os
import numpy as N
from casa_config import *

print "Will use flagants and flagscans if present"
print "Using", msfile
print "Refant", myref
print "Gaincal chans", gaincal_bchan, gaincal_echan
print "Do bandpass every", bp

if removewt:
  tb.open(msfile, nomodify=False)
  tb.removecols('WEIGHT_SPECTRUM')
  tb.close()

flagants = []
if os.path.isfile('flagants'):
  dd = libs.readinfile('flagants')[0]
  flagants = list(dd)

flagscans = []
if os.path.isfile('flagscans'):
  dd = libs.readinfile('flagscans')[0]
  flagscans = list(N.asarray(dd,int))

if len(flagants)>0:
    default('flagcmd')
    vis = msfile
    outfile = ''
    for ant in flagants:
        dum = "antenna='"+str(ant)+"'"
        inpfile = [dum]
        print "Flagging antennas ", inpfile
        flagcmd(vis=vis,savepars=False,outfile=outfile,action='apply',\
                inpmode='list',inpfile=inpfile,flagbackup=False)
    
if len(flagscans)>0:
    default('flagcmd')
    vis = msfile
    outfile = ''
    for scan in flagscans:
        dum = "scan='"+str(scan)+"'"
        inpfile = [dum]
        print "Flagging scans ", inpfile
        flagcmd(vis=vis,savepars=False,outfile=outfile,action='apply',\
                inpmode='list',inpfile=inpfile,flagbackup=False)

# GAIN CAL
default('gaincal')
vis = msfile
spw = '0:'+str(gaincal_bchan)+'~'+str(gaincal_echan)
field = ''
solint='8s'
gaintable = ''
refant = myref
caltable = 'sn_table_int_ref'+myref+'_'+str(gaincal_bchan)+'_'+str(gaincal_echan)
#gaincal(vis=vis,field=field,caltable=caltable,gaintable=gaintable,solint=solint,refant=refant,spw=spw)

# BANDPASS CALIBRATION
default('bandpass')
vis = msfile
field = ''
combine = ''
refant = ''
solnorm = False
if bp=='scan':
  solint = 'Inf'
  caltable = 'bandpass_table_scan'
else:
  solint = '8s'
  caltable = 'bandpass_table_int'
bandpass(vis=vis,field=field,caltable=caltable,combine=combine,solint=solint,refant=refant,solnorm=solnorm)


