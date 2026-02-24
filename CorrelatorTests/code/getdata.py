#!/usr/bin/env python
import os, sys, libs
import numpy as N
import katdal

basedir = '/data/mohan/'
basedir = os.getcwd()
fname = sys.argv[1]

if fname == '-h':
  print("USAGE")
  print("./getdata.py NUMBER NAME simple/full")
  print("or")
  print("./getdata list FILENAME (which has list of num and name)")
  exit()

if fname == 'list':
  filename = sys.argv[2]
  if not os.path.isfile(basedir+filename):
    raise RuntimeError(basedir+filename+" not found")
  dd = libs.readinfile(basedir+filename)
  #allnums = N.asarray([num.split('.')[0] for num in dd[0]])
  allnums = N.asarray(dd[0], int)
  allnames = N.asarray(dd[1])
else:
  allnums = [fname]
  allnames = [sys.argv[2]]
nf = len(allnums)

for ii in range(nf):
  print("Doing ", allnums[ii], allnames[ii])
  fname = allnums[ii]
  if os.path.isdir(basedir+fname):
    print("Directory already exists; will overwrite file")
    os.chdir(basedir+fname)
  else:
    os.mkdir(basedir+fname)
  
  if fname == 'list': dextn = 'simple'
  else: dextn = sys.argv[3]
  if dextn=='simple': extn = '.rdb'
  if dextn=='full': extn = '.full.rdb'
  str1 = "wget -P " + basedir+fname+" http://archive-gw-1.kat.ac.za:7480/"+fname+"/"+fname+"_sdp_l0"+extn
  os.chdir(basedir+fname)
  os.system(str1)
  
  name = allnames[ii]
  if ' ' in name: 
      raise RuntimeError("Name should not have spaces")

  os.system('touch name='+name)
  f = katdal.open(fname+"_sdp_l0"+extn)

  orig_stdout = sys.stdout
  fn = open('printf', 'w')
  sys.stdout = fn
  str1 = str(f)
  print(str1)
  sys.stdout = orig_stdout
  fn.close() 
  
  str1 = "cd "+basedir+"; ln -s "+ fname + " " + name
  print(str1)
  os.system(str1)
  
