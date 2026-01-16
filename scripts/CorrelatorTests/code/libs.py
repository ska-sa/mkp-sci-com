

def readinfile(fname, skipbeg=0, skipend=0, array=None):
    """ Read the rows in fname and returns a column-wise list. If the column is a 
    number then it is a numpy array, else a list of strings. If array is a type and all
    members are of that type, then it returns a 2d array of that type. It ignores skip number 
    of lines at the beginning and end. Default skip=0"""
    import numpy as N
    import os

    if not os.path.isfile(fname): 
      raise RuntimeError("File "+fname+" not found")
    f=open(fname, 'r')
    dlist = []
    for i,line in enumerate(f):
        if i >= skipbeg:
          dlist.append(line.split())
    f.close()
    if skipend > 0: 
      dlist = dlist[:-skipend]

    try:
      new= [[r[col] for r in dlist] for col in range(len(dlist[0]))]
    except:
      raise RuntimeError("Problem reading File "+fname)
      

    if array == None:
      dlist = []
      for col in new:
        ele = col[0]
        try:
          x=float(ele)
          dlist.append(N.asarray(col, dtype=float))
        except:
          dlist.append(col)
      return dlist
    else:
      new = N.array(new, dtype=array)
      return new

def readinfile_known(fname, shape, thetype=None, skipbeg=0, skipend=0, ignorecomments=False):
    """ The file is made of floats, and the array shape is known beforehand. Useful for when the 
    datasize is so huge that readinfile takes way too much memory
    If ignorecomments is False then ignore lines starting with # after skipbeg else crash if it is there"""

    import numpy as N

    if thetype==None: thetype=float
    if isinstance(shape, int):
      n = shape
      arr = N.zeros(n-skipend, dtype=thetype)
    else:
      n, m = shape
      arr = N.zeros((n-skipend,m), dtype=thetype)

    f = open(fname, 'r')
    for i,line in enumerate(f):
      if i >= skipbeg-1: break

    if thetype==None: thetype=float
    # if need to check for comment
    if ignorecomments:
      for i,line in enumerate(f):
        if i<n-skipend:
          if line[0] != '#':
            arr[i] = N.asarray(line.split(), dtype=thetype)

    # if no need to check for comment
    if isinstance(shape, int):
      for i,line in enumerate(f):
        if i<n-skipend: arr[i] = thetype(line)
    else:
      for i,line in enumerate(f):
        if i<n-skipend: arr[i] = N.asarray(line.split(), dtype=thetype)

    f.close()
    return arr

             
def convert_time(x):
    
    if isinstance(x, float):
      h = int(x)
      dum = (x-h)*60.0
      m = int(dum)
      s = (dum - m)*60.0
      d = h, m, s
    else :
      try:
        if len(x) == 3:
          d = x[0] + x[1]/60.0 + x[2]/3600.0
      except:
        print('Dont be silly')
        raise RuntimeError

    return d

def justdist(ra1,ra2,dec1,dec2):
    """ Computes distance on surface of a sphere in arcsec 
    between (ra1,dec1) and (ra2,dec2) in degrees. """
    from math import pi,sin,cos,acos

    rad=pi/180.0

    r1=(ra1-180.0)*rad
    r2=(ra2-180.0)*rad
    d1=dec1*rad
    d2=dec2*rad
    dum = sin(d1)*sin(d2)+cos(d1)*cos(d2)*cos(r1-r2)
    if dum>1.0: dum = 1.0 # rounding off error for zero dist
    dist=acos(dum)*3600.0/rad  

    return dist

# Fake an ellipse using an N-sided polygon
def Ellipse(cens, ras, alpha, nbin=50, **kwargs):
    import numpy as N
    from math import pi,sin,cos
    import pylab as pl

    cenx, ceny = cens
    radx, rady = ras
    theta = 2*pi/nbin*N.arange(nbin) 
    xs = cenx + radx * N.cos(theta)
    ys = ceny + rady * N.sin(theta)

    return pl.Polygon(zip(xs, ys), closed=True, alpha=alpha, **kwargs)

def convertcoords(params, ctype=None):
    """ 
    Convert ra, dec in deg to hh,mm etc and vice versa. Input is a list.
    Hence len(params) should be 2 or 6.
    if ctype = 'altaz' then assume altaz (no 15) else radec (x/15)
    """
    import numpy as N

    if len(params) not in [2, 6]:
      print("convertcoords: Coordinates dont make sense")
    else:
      if ctype not in ['radec', 'altaz']: ctype = 'radec'
      if ctype == 'radec':
        fac = 15.0
      else:
        fac = 1.0

      if len(params) == 2:
        ra, dec = params 
        dumr = ra/fac
        hh = int(dumr); dum = (dumr - hh)*60.0
        mm = int(dum); ss = (dum - mm)*60.0
        sgn = 1
        if dec < 0: sgn = -1
        dumr = abs(dec)
        dd = int(dumr); dum = (dumr - dd)*60.0
        ma = int(dum); sa = (dum - ma)*60.0
        return hh, mm, ss, sgn, dd, ma, sa
      if len(params) == 6:
        hh, mm, ss, dd, ma, sa = params
        sgn = 1
        if dd < 0 or ma < 0 or sa < 0: sgn = -1
        ra = (hh+mm/60.0+ss/3600.0)*fac
        dec = sgn*(abs(dd)+abs(ma)/60.0+abs(sa)/3600.0)
        return ra, dec

def B1950toJ2000(Bcoord):
    """ Precess using Aoki et al. 1983. Same results as NED to ~0.2asec """
    from math import sin, cos, pi, sqrt, asin, acos
    import numpy as N

    rad = 180.0/pi
    ra, dec = Bcoord

    A = N.array([-1.62557e-6, -0.31919e-6, -0.13843e-6])
    M = N.array([[0.9999256782, 0.0111820609, 0.00485794], [-0.0111820610, 0.9999374784, -0.0000271474], \
                 [-0.0048579477, -0.0000271765, 0.9999881997]])

    r0=N.zeros(3)
    r0[0]=cos(dec/rad)*cos(ra/rad)
    r0[1]=cos(dec/rad)*sin(ra/rad)
    r0[2]=sin(dec/rad)

    r0A=N.sum(r0*A)
    r1=r0-A+r0A*r0
    r = N.sum(M.transpose()*r1, axis = 1)

    rscal = sqrt(N.sum(r*r))
    decj=asin(r[2]/rscal)*rad 

    d1=r[0]/rscal/cos(decj/rad)
    d2=r[1]/rscal/cos(decj/rad)
    raj=acos(d1)*rad 
    if d2 < 0.0: raj = 360.0 - raj

    Jcoord = [raj, decj]
    return Jcoord


def add_fits(inp, out, dir=None):

   """ Add all fits files in list inp if they all have the same size. Output (name = out) has the header of the first file. 
   Does not check for blanks for now. """
   import numpy as N
   import pyfits 
   import os

   if isinstance(inp, list) and isinstance(out, str) and N.all([isinstance(ii, str) for ii in inp]):
     if dir == None: dir = ''
     for ifile, fitsf in enumerate(inp):
       fitsname = dir+fitsf
       if not os.path.isfile(fitsname):
         raise ValueError(fitsname + ' does not exist')
       fits = pyfits.open(fitsname)
       if ifile == 0:
         header = fits[0].header
         data = fits[0].data
       else:
         if data.shape != fits[0].data.shape:
           raise ValueError(fitsname + 'has the wrong shape')
         data = data + fits[0].data
     pyfits.writeto(dir+out, data, header, clobber=True)
   else:
     raise ValueError("First input should be a list of strings and second should be a string")


def comb_fits(inp, out, op, dir=None):

   """ Add all fits files in list inp if they all have the same size. Output (name = out) has the header of the first file. 
   Does not check for blanks for now. op is 'add/sub/mul/div'"""
   import pyfits 
   import os

   if isinstance(inp, list) and isinstance(out, str) and N.all([isinstance(ii, str) for ii in inp]):
     if op in ['add', 'sub', 'mul', 'div']:
       if dir == None: dir = ''
       for ifile, fitsf in enumerate(inp):
         fitsname = dir+fitsf
         if not os.path.isfile(fitsname):
           raise ValueError(fitsname + ' does not exist')
         fits = pyfits.open(fitsname)
         if ifile == 0:
           header = fits[0].header
           data = fits[0].data
         else:
           if data.squeeze().shape != fits[0].data.squeeze().shape:
             raise ValueError(fitsname + 'has the wrong shape')
           if op == 'add': data = data + fits[0].data
           if op == 'sub': data = data - fits[0].data
           if op == 'mul': data = data * fits[0].data
           if op == 'div': data = data / fits[0].data
       ### FOR THIS WEIRD FILE A2255_85CM_BEAM_cut.fits
       header['NAXIS'] = 3
       header['BITPIX'] = -32
       pyfits.writeto(dir+out, data, header, output_verify='ignore', clobber=True)
   else:
     raise ValueError("First input should be a list of strings and second should be a string")


def math_fits(inp, out, f_op, dir=None):

   """ Performs f_op function on inp and saves it to out. """
   import pyfits 
   import os

   if isinstance(inp, str) and isinstance(out, str):
     if dir == None: dir = ''
     fitsname = dir+inp
     if not os.path.isfile(fitsname):
       raise ValueError(fitsname + ' does not exist')
     fits = pyfits.open(fitsname)
     header = fits[0].header
     data = fits[0].data
     data = f_op(data)
     pyfits.writeto(dir+out, data, header, clobber=True)
   else:
     raise ValueError("First input should be a list of strings and second should be a string")


def change_srt(fn, shift):

  file1 = open(fn, 'r')
  file2 = open(fn[:-4]+'_mod.srt', 'w')
  
  ctr = 0 
  for line in file1:
    ctr += 1
    if (':' not in line) or (len(line.split(':')) != 5):
      file2.write(line)
    else:
      try:
        a = line.split(':')
        h1, m1, s1, m2, s2 = a
      except: 
        print(ctr, line)
        raise RuntimeError() 
      h1 = int(h1)
      m1 = int(m1)
      m2 = int(m2)
      a1, h2 = s1.split('-->')
      s11, s12 = a1.split(',')
      s1 = float(s11)+float(s12)*0.001
      h2 = int(h2)
      s21, s22 = s2.split(',')
      s2 = float(s21)+float(s22)*0.001
  
      s1 = s1+shift
      if s1 >= 60: 
        s1 -= 60
        m1 += 1
      if s1 < 0: 
        s1 += 60
        m1 -= 1
      s2 = s2+shift
      if s2 >= 60: 
        s2 -= 60
        m2 += 1
      if s2 < 0: 
        s2 += 60
        m2 -= 1
      str1 = str(h1)+':'+str(m1)+':'+str(s1)+' --> '+str(h2)+':'+str(m2)+':'+str(s2)+'\n'
      file2.write(str1)
  
  file1.close()
  file2.close()

def randcomp(size, rtype, norm=False):
    import numpy as N
    import math

    if rtype not in ['normal', 'random']:
      raise RuntimeError

    if rtype == 'normal':
      r_amp = N.random.normal(0.0, 1.0, size)
    if rtype == 'random':
      r_amp = N.random.random(size)
    r_angle = N.random.random(size)*360.0*math.pi/180.0

    if norm==True:
      r_amp = N.ones(size)

    r_re = r_amp*N.cos(r_angle)
    r_im = r_amp*N.sin(r_angle)
    
    return r_re + r_im * 1j

def mymkdir(solnname, mode=None):
    import os

    if mode not in ['del', 'keep']: mode = 'del'
    if os.path.exists(solnname):
      if os.path.isfile(solnname):
        os.system('rm -f '+solnname)
        os.mkdir(solnname)
      else:
        if os.path.isdir(solnname):
          os.system('rm -fr '+solnname+'/*')
        else:
          raise RuntimeError("Something wrong with creating "+solnname)
    else:
      os.mkdir(solnname)
 
def pplot(x, tit, ii, func):
    import math
    import pylab as pl

    ii += 1
    y = func(x)
    pl.figure(i)
    pl.subplot(3,3,ii)
    pl.imshow(y, interpolation='nearest', origin='lower')
    pl.colorbar()
    pl.title(tit)

    return ii

def stats1d(arr):
    import numpy as N

    d = {}
    d['max'] = N.max(arr)
    d['min'] = N.min(arr)
    d['mean'] = N.mean(arr)
    d['median'] = N.median(arr)
    d['std'] = N.std(arr)

    return d

def isinvsym(a, centre=None):
    """ Checks if arr is inversion symmetric. If centre is None, calculates peak as centre. 
    """
    import numpy as N

    arr = N.copy(a)
    if centre==None: 
      n, m = N.unravel_index(N.argmax(arr), arr.shape)
    else:
      n, m = centre
    xx = min(n, arr.shape[0]-1-n)
    yy = min(m, arr.shape[1]-1-m)

    arr1 = arr[n-xx:n+xx+1,m-yy:m+yy+1]
    nn, mm = arr1.shape
    n, m = N.unravel_index(N.argmax(arr1), arr1.shape)

    for row in range(m+1,mm):
      arr1[:,row] -= arr1[:,2*m-row][::-1]
    arr1[:n,m] -= arr1[n+1:,m][::-1]

    arr = arr1[:,m+1:].flatten()
    arr = N.concatenate((arr, arr1[:n,m]))

    return stats1d(arr) 

def gaus_1d(c, x):
    import numpy as N

    if len(c) == 4:
      y = c[0]*N.exp(-0.5*(x-c[1])*(x-c[1])/(c[2]*c[2]))+c[3]
    if len(c) == 3:
      y = c[0]*N.exp(-0.5*(x-c[1])*(x-c[1])/(c[2]*c[2]))

    return y

def poly_n(c, x, n):
    """ Sum c_i * x_i^i """
    import numpy as N

    y = c[0]
    for i in range(n):
      y = y + c[i+1]*pow(x, i+1)
    return y 

def decay1(c, x):
    import numpy as N

    y = c[0]+ c[1]*N.exp(N.power(x, c[2]))
    return y

def decay2(c, x):
    import numpy as N

    y = c[0]+ c[1]*N.exp(-N.power(x, 4)/c[2])
    return y

def fit_1d(x, y, sig, func, c=None):
    from scipy.optimize import leastsq
    import numpy as N

    if c==None:
      c = [N.max(y), N.argmax(y), 0.62]
    res = lambda p, x, yfit: (yfit-func(p,x))/sig
    p = leastsq(res, c, args=(x, y))[0]


    return p

def poshfit_1d(x, y, func, sig=None, mask=None, do_err=False, p0 = None):
    """ 
    Calls scipy.optimise.leastsq for a 1d function with a mask and many options.
    Takes values only where mask=False.
    """
    from scipy.optimize import leastsq
    import numpy as N
    from math import sqrt

    if len(x) != len(y):
      print('Inputs x and y for fit should be of same length')
      raise RuntimeError

    # mask
    if (isinstance(mask, None.__class__)) and (mask == None): mask = N.zeros(x.shape)
    mask = N.asarray(mask, dtype=bool)
    ind=N.where(~mask)[0]
    if len(ind) > 1: xfit=x[ind]; yfit=y[ind] 

    # sigma
    if sig == None: sigfit = N.ones(xfit.shape) 
    if sig == 'calc': sigfit = N.ones(xfit.shape) 
    if isinstance(sig, x.__class__):
      if len(sig) == len(x):
        sigfit=sig[ind]
      else:
        print('Something wrong with sigma for fit')
        raise RuntimeError

    if len(ind) > 2:
      # p0
      if func == gaus_1d:
        if p0 == None or (p0 != None and len(p0) != 3):
          p0 = [N.max(yfit), xfit[N.argmax(yfit)], 4.0, N.min(yfit)]
        res=lambda p, xfit, yfit, sigfit: (yfit-func(p, xfit))/sigfit

      if func == poly_n:
        npoly = len(p0)-1
        res=lambda p, xfit, yfit, sigfit: (yfit-func(p, xfit, npoly))/sigfit
 
      (p, cov, info, mesg, flag)=leastsq(res, p0, args=(xfit, yfit, sigfit), full_output=True)
 
      if sig =='calc':
        sigfit = N.ones(xfit.shape)*N.std(yfit-func(p, xfit))
        (p, cov, info, mesg, flag)=leastsq(res, p0, args=(xfit, yfit, sigfit), full_output=True)
   
      if do_err: 
        if cov != None:
          if N.sum(sig != 1.) > 0:
            err = N.array([sqrt(cov[i,i]) for i in range(len(p))])
          else:
            chisq=sum(info["fvec"]*info["fvec"])
            dof=len(info["fvec"])-len(p)
            err = N.array([sqrt(cov[i,i]*chisq/dof) for i in range(len(p))])
        else:
          p, err = [0, 0], [0, 0]
      else: err = [0]
    else:
      p, err = [0, 0], [0, 0]
 
    return p, err

def subplot(x):
    import math 
    
    n=int(round((math.sqrt(x))))
    m = int(math.ceil(x/n))
    if n*m < x: m+=1

    return n, m

def wc(fname):
    import os, commands

    if not os.path.isfile(fname):
      raise RuntimeError("Cannot find "+fname)
    else:
      nn = int(commands.getoutput('wc -l '+fname).split()[0])

    return nn
 
def hist_med(d, n, rnge='none', offset=0, mtype='full', fac=1,ver=0):
  import numpy as N

  low = -n/2-offset
  high = n/2-offset
  if rnge=='calc':
    low = N.min(d); high = N.max(d)
  x = N.arange(low*fac, high*fac)
  y = N.zeros(n*fac)
  d = d*fac
  nd = len(d)
  for data in d:
    y[data-low] += 1
 
  if nd%2==1:
    val = (nd+1)/2
  else:
    if mtype in ['low', 'full']:  
      val = nd/2
    if mtype=='high':  
      val = nd/2+1

  s=N.zeros(n*fac); s[0] = y[0]
  for i in range(1,len(y)):
    s[i] = s[i-1]+y[i]
    if s[i] > nd/2:
      break

  v1 = N.searchsorted(s[:i+1], val)
  v2 = N.searchsorted(s[:i+1], val+(1-nd%2))
  if mtype in ['low', 'high']: 
    med = x[v1]
  if mtype == 'full': 
    med = 0.5*(x[v1]+x[v2])

  med = med/fac

  return med


def med_from_hist_int(hist, x, mtype='full'):
  import numpy as N

  nd = N.sum(hist)
  n = len(hist)
  if nd%2==1:
    val = (nd+1)/2
  else:
    if mtype in ['low', 'full']:  
      val = nd/2
    if mtype=='high':  
      val = nd/2+1

  s=N.zeros(n); s[0] = hist[0]
  for i in range(1,len(hist)):
    s[i] = s[i-1]+hist[i]
    if s[i] > nd/2:
      break

  v1 = N.searchsorted(s[:i+1], val)
  v2 = N.searchsorted(s[:i+1], val+(1-nd%2))
  if mtype in ['low', 'high']: 
    med = x[v1]
  if mtype == 'full': 
    med = 0.5*(x[v1]+x[v2])

  return med

def nanstd(x, n=None):
  """ Do biased estimator"""
  import numpy as N

  if n==None:
    mean = nanmean(x)
    a1 = mean*mean
    a2 = nanmean(x*x)
  else:
    mean = N.nansum(x, n)/x.shape[n]
    a1 = mean*mean
    a2 = N.nansum(x*x, n)/x.shape[n]

  return N.sqrt(a2-a1)

  
def nanmean(x):
    """ Mean of array with NaN """
    import numpy as N

    sum = N.nansum(x)
    n = N.sum(~N.isnan(x))

    if n > 0:
      mean = sum/n
    else:
      mean = float("NaN")

    return mean

def nanmedian(x):
    import numpy as N

    return N.median(x[N.where(~N.isnan(x))])

def mad(x, n=None):
    """MAD of array x"""
    import numpy as N

    if n==None:
      med = N.median(x)
      mad = N.median(N.abs(x-med))
    else:
      if len(x.shape) != 2:
        raise RuntimeError("ERROR : nammad only implemented for 1d and 2d arrays")
      med = N.median(x, n)
      y = N.copy(x)
      if n==0:
        for i in range(x.shape[n]):
          y[i] = x[i]-med
        mad = N.median(N.abs(y), n)
      if n==1:
        for i in range(x.shape[n]):
          y[:,i] = x[:,i]-med
        mad = N.median(N.abs(y), n)
    
    return mad

def imshow(im, scale=False, vmin=None, vmax=None, extent=None):
    import pylab as pl
    import numpy as N

    if scale:
      med = N.median(im)
      mad = 1.5*N.median(N.abs(im-med))
      vmin = med-15*mad
      vmax = med+15*mad

    if extent==None:
      pl.imshow(N.transpose(im), origin='lower', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
    else:
      pl.imshow(N.transpose(im), origin='lower', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)

def fmt(x, n):
  import math as m

  y = round(x*m.pow(10,n))/pow(10,n)
  
  return y


