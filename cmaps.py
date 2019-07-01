import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def converto2Dmap(data):
  '''
  data is 1D array containing values on each phi/psi combination
  '''
  l=[]
  data = np.asarray(data)
  for n in data:
    n = "{:9.5f}".format(float(n))
    l.append(float(n))
  l = np.asarray(l)
  shape=(24, 24)
  l = l.reshape(shape)
  return l

def converto2Dmap0(data, n_zero):
  '''
  data is 1D array containing values on each phi/psi combination, n_zero is the index for the zero point, check the index datafile for more information
  '''
  zero = float(data[n_zero])
  data = np.asarray(data)
  data[:] = [float(x) - zero for x in data]
  l=[]
  for n in data:
    n = "{:9.5f}".format(float(n))
    l.append(float(n))
  l = np.asarray(l)
  shape=(24, 24)
  l = l.reshape(shape)
  return l

def converto2Dmap0_finegrid(data, n_zero):
  '''
  data is 1D array containing values on each phi/psi combination, n_zero is the index for the zero point, check the index datafile for more information
  '''
  zero = float(data[n_zero])
  data = np.asarray(data)
  data[:] = [float(x) - zero for x in data]
  l=[]
  for n in data:
    n = "{:9.5f}".format(float(n))
    l.append(float(n))
  l = np.asarray(l)
  shape=(72, 72)
  l = l.reshape(shape)
  return l

def converto2Dmap0sizeN(data, n_zero, size):
  '''
  data is 1D array containing values on each phi/psi combination, n_zero is the index for the zero point, check the index datafile for more information
  '''
  zero = float(data[n_zero]) 
  data = np.asarray(data)
  data[:] = [float(x) - zero for x in data]
  l=[]
  for n in data:
    n = "{:9.5f}".format(float(n))
    l.append(float(n))
  l = np.asarray(l)
  shape=(size, size)
  l = l.reshape(shape)
  return l

def savetoFRCMOD(data, filename):
  data = np.asarray(data)
  data.resize((72,8))
  np.savetxt(filename, data, fmt='%9.5f')

def pairwisediff(m1, m2, x1, x2, x3, x4, y1, y2, y3, y4): 
  '''
  m1 m2 are 2D matrix; x, y are index, not dihedral
  '''
  diff = []
  region1 = []
  region2 = []
  #define the regions
  for i1 in range(x1,x2):
    for j1 in range(y1,y2):
      region1.append(m1[i1][j1])
      region2.append(m2[i1][j1])
  for i1 in range(x3,x4):
    for j1 in range(y3,y4):
      region1.append(m1[i1][j1])
      region2.append(m2[i1][j1])
  #iterate over the regions
  for a in range(len(region1)):
    for b in range(len(region1)):
      x = abs((float(region1[a])-float(region1[b]))-(float(region2[a])-float(region2[b])))
      diff.append(x)
  #remove zero values
  for element in diff:
    if int(element) == 0:
      diff.remove(element)
  avg_diff = np.average(diff)
  return avg_diff

def subplotheatmap30degree(data,title,range=None,vmin=None, vmax=None, ax=None):
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")
  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  heatmap = data.transpose()
  #heatmap = np.flipud(heatmap)
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('w')
  cmap.set_under('w')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    if range == None:
      b = float(a) + 12.0
    else:
      b = float(a) + float(range)
    im = ax.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = ax.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = plt.colorbar(im,ax=ax)
  cbar.ax.tick_params(labelsize=14)
  tick = np.arange(0, 12, 1.8)
  ax.set_xticks(tick)
  ax.set_xticklabels(['-180','-120','-60','0','60','120','180'],fontsize=14)
  ax.set_yticks(tick)
  ax.set_yticklabels(['-180','-120','-60','0','60','120','180'],fontsize=14)
  ax.set_xlim(0,11)
  ax.set_ylim(0,11)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  ax.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  ax.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  ax.set_title(title,fontsize=20)

def subplotheatmap(data,title,range=None,vmin=None, vmax=None, ax=None):
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")
  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  heatmap = data.transpose()
  #heatmap = np.flipud(heatmap)
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('#67000d')
  cmap.set_under('#3f007d')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    if range == None:
      b = float(a) + 12.0
    else:
      b = float(a) + float(range)
    im = ax.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = ax.imshow(heatmap,interpolation='none',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = plt.colorbar(im,ax=ax)
  cbar.ax.tick_params(labelsize=8)
  tick = np.arange(0, 24, 3.8)
  ax.set_xticks(tick)
  ax.set_xticklabels(['-180','-120','-60','0','60','120','180'],fontsize=8)
  ax.set_yticks(tick)
  ax.set_yticklabels(['-180','-120','-60','0','60','120','180'],fontsize=8)
  ax.set_xlim(0,23)
  ax.set_ylim(0,23)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  #ax.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  #ax.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  ax.set_title(title,fontsize=10)

def plotheatmap(data, title, range=None,vmin=None, vmax=None):
  '''
  data is 2D array containing energy values for phi/psi; vmin and vmax is the range of colorbar, when None, use the mininal in  the dataset, and min+15.0 as the maximum.
  '''
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")

  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  fig = plt.figure(figsize=(8,6))
  heatmap = data.transpose()
  #heatmap = np.flipud(heatmap)
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('w')
  cmap.set_under('w')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    if range == None:
      b = float(a) + 12.0
    else:
      b = float(a) + float(range)
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = plt.colorbar(im)
  cbar.ax.tick_params(labelsize=14)
  tick = np.arange(0, 24, 3.8)
  plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
  plt.yticks(tick,['-180','-120','-60','0','60','120','180'],rotation='horizontal')
  plt.tick_params(labelsize=14)
  plt.xlim(0,23)
  plt.ylim(0,23)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  plt.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  plt.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  plt.title(title,fontsize=20)
  plt.savefig(title+'.png',dpi=300)

def plotheatmap_finegrid(data, title, range=None,vmin=None, vmax=None):
  '''
  data is 2D array containing energy values for phi/psi; vmin and vmax is the range of colorbar, when None, use the mininal in  the dataset, and min+15.0 as the maximum.
  '''
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")

  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  fig = plt.figure(figsize=(8,6))
  heatmap = data.transpose()
  #heatmap = np.flipud(heatmap)
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('w')
  cmap.set_under('w')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    if range == None:
      b = float(a) + 12.0
    else:
      b = float(a) + float(range)
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = plt.colorbar(im)
  cbar.ax.tick_params(labelsize=14)
  tick = np.arange(0, 72, 11.8)
  plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
  plt.yticks(tick,['-180','-120','-60','0','60','120','180'],rotation='horizontal')
  plt.tick_params(labelsize=14)
  plt.xlim(0,71)
  plt.ylim(0,71)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  plt.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  plt.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  plt.title(title,fontsize=20)
  plt.savefig(title+'.png',dpi=300)

def plotheatmapfinegrid(data, title, range=None,vmin=None, vmax=None):
  '''
  data is 2D array containing energy values for phi/psi; vmin and vmax is the range of colorbar, when None, use the mininal in  the dataset, and min+15.0 as the maximum.
  '''
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")

  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  fig = plt.figure(figsize=(8,6))
  heatmap = data.transpose()
  #heatmap = np.flipud(heatmap)
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('w')
  cmap.set_under('w')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    if range == None:
      b = float(a) + 12.0
    else:
      b = float(a) + float(range)
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = plt.imshow(heatmap,interpolation='none',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = plt.colorbar(im)
  cbar.ax.tick_params(labelsize=14)
  tick = np.arange(0, 70, 11.5)
  plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
  plt.yticks(tick,['-180','-120','-60','0','60','120','180'],rotation='horizontal')
  plt.tick_params(labelsize=14)
  plt.xlim(0,69)
  plt.ylim(0,69)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  plt.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  plt.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  plt.title(title,fontsize=20)
  plt.savefig(title+'.png',dpi=300)

def plotpmf(x,y, title, vmin=None, vmax=None):
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import seaborn as sns; sns.set(style="ticks")
  from math import log
  
  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  fig = plt.figure(figsize=(8,6))
  xedges, yedges = np.linspace(-180, 180, 72), np.linspace(-180, 180, 72)
  H,X,Y = np.histogram2d(y,x,(xedges, yedges))
  M=int(max(H.max(axis=1)))
  for x in np.nditer(H, op_flags=['readwrite']):
    pop=float(x/M)
    if pop==0.0:
      x[...]=10.0
    else:
      x[...]=-0.59219*log(pop)
  heatmap = H
  c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
  color=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
  cmap = mpl.colors.ListedColormap(sns.color_palette(color, 20))
  cmap.set_over('white')
  cmap.set_under('#3f007d')
  if vmin == None and vmax == None:
    a = float(np.amin(heatmap))
    b = float(a) + 15.0
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=a, vmax=b))
  else:
    im = plt.imshow(heatmap,interpolation='bicubic',cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
  cbar = sns.plt.colorbar(im)
  cbar.ax.tick_params(labelsize=10)
  tick = np.arange(0, 72, 11.7)
  plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
  plt.yticks(tick,['-180','-120','-60','0','60','120','180'],rotation='horizontal')
  plt.tick_params(labelsize=15)
  plt.xlim(0,71)
  plt.ylim(0,71)
  l=[]
  l2=[]
  if vmin == None and vmax == None:
    i=int(a)
    j=float(i) + 0.5
    while i < float(b)+1.0:
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  else:
    i=vmin
    j=float(vmin)+0.5
    while i < float(vmax):
      l.append(i)
      l2.append(j)
      i = i + 1
      j = j + 1.0
  levels = np.asarray(l)
  levels2 = np.asarray(l2)
  plt.contour(heatmap,levels, colors='black',linestyles='solid',linewidths=0.8)
  plt.contour(heatmap,levels2, colors='black',linestyles='dashed',linewidths=0.8)
  #plt.title(title,fontsize=30)
  plt.savefig(title+'.png',dpi=300)

def plotsubheatmap(m,n,title):
  '''
  m is row, n is column of subplots, this is for heatmap subplot
  '''
  import matplotlib.pyplot as plt
  import seaborn as sns; sns.set(style="ticks")
  import numpy as np
  import matplotlib as mpl
  from numpy import ma
  import matplotlib.colors as colors
  from matplotlib.colors import LinearSegmentedColormap
  import sys
 
  sns.set_style({"xtick.direction": "in","ytick.direction": "in" },{"xtick.major.size": 4, "ytick.major.size": 4})
  fig,axes = plt.subplots(m,n,figsize=(4*m,4*n))
  axes[-1, -1].axis('off')
  plt.subplots_adjust(wspace=0.2,hspace=0.3)
  lim = n*m + 1
  for i in range(1,lim):
    ax = plt.subplot(m,n,i)
    heatmap = np.loadtxt(sys.argv[i],delimiter=',')
    heatmap = heatmap.transpose()
    c_norm = mpl.colors.BoundaryNorm(boundaries=[-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],ncolors=20)
    kbcolor=['#3f007d','#54278f','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#ffffe5','#ffffe5','#fff5eb','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d'] #colorbrewer two purple, seven blue, three yellow, #7 eight red
    cmap = mpl.colors.ListedColormap(sns.color_palette(kbcolor, 20))
    min = float(np.amin(heatmap))
    ma = float(min) + 15.0
    im = ax.imshow(np.flipud(heatmap),interpolation='bicubic',cmap=cmap, norm=colors.Normalize(vmin=min, vmax=ma))
    cbar = sns.plt.colorbar(im)
    cmap.set_over('w')
    cmap.set_under('w')
    cbar.ax.tick_params(labelsize=10)
    tick = np.arange(0, 24, 3.8)
    plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
    plt.yticks(tick,['180','120','60','0','-60','-120','-180'],rotation='horizontal')
    plt.tick_params(labelsize=10)
    plt.grid(False)
    l=[]
    l2=[]
    i=int(min)
    j=float(i) + 0.5
    while i < float(ma)+1.0:
      l.append(i)
      l2.append(j)
    i = i + 1
    j = j + 1.0
    levels = np.asarray(l)
    levels2 = np.asarray(l2)
    cs = ax.contour(np.flipud(heatmap),levels, colors='black',linestyles='solid',linewidths=0.8)
    cs = ax.contour(np.flipud(heatmap),levels2, colors='black',linestyles='dashed',linewidths=0.8)
    plt.clabel(cs,fontsize=0)
    plt.title(sys.argv[i],fontsize=14)
  plt.suptitle(title,fontsize=16)
  plt.savefig(title+'.png',dpi=300)

def plotrama(x,y,title,N=None,subtitle=None,ylabel=None,chi2=None,std_chi2=None,hp=None,std_hp=None,pp2=None,std_pp2=None,beta=None,std_beta=None,box=None):
  '''
  x and y is 1D array, i.e. dihedral values from MD trajectory
  '''
  import matplotlib.pyplot as plt
  import numpy as np
  import math
  from matplotlib.ticker import NullFormatter
  from matplotlib import colors
 
  nullfmt = NullFormatter()

  left, width = 0.23, 0.63
  bottom, height = 0.15, 0.63
  bottom_h = bottom + height
  left_h = width + left

  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom_h, width, 0.1]
  rect_histy = [left_h, bottom, 0.1, height]

  fig = plt.figure(figsize=(3, 3))
  fig.suptitle(subtitle,fontsize=12)
  axScatter = plt.axes(rect_scatter)
  axScatter.set_xlim((0,71))
  axScatter.set_ylim((0,71))
  axScatter.set_ylabel(ylabel,fontsize=12)
  tick = np.arange(0, 72, 11.8)
  plt.xticks(tick,['-180','-120','-60','0','60','120','180'])
  plt.yticks(tick,['-180','-120','-60','0','60','120','180'],rotation='horizontal')
  plt.tick_params(labelsize=10)
  axHistx = plt.axes(rect_histx)
  axHisty = plt.axes(rect_histy)
  axHistx.set_xlim((-180,180))
  axHisty.set_ylim((-180,180))
  axHistx.xaxis.set_major_formatter(nullfmt)
  axHisty.yaxis.set_major_formatter(nullfmt)
  axHistx.xaxis.set_visible(False)
  axHistx.yaxis.set_visible(False)
  axHisty.xaxis.set_visible(False)
  axHisty.yaxis.set_visible(False)
  axHistx.axis('off')
  axHisty.axis('off')
  #plot contour map and 1D hist
  xedges, yedges = np.linspace(-180, 180, 72), np.linspace(-180, 180, 72)
  H,X,Y = np.histogram2d(y,x,(xedges, yedges))
  M = [max(sub_array) for sub_array in H]
  ma = int(max(M))
  im = axScatter.imshow(H,cmap="Purples",interpolation="None",norm=colors.Normalize(vmin=1, vmax=ma))
  axScatter.set_xlim((0, 72))
  axScatter.set_ylim((0, 72))
  bins = np.arange(-180, 181, 5)
  axHistx.hist(x, bins=bins,color='#bcbddc',linewidth=0.4)
  axHisty.hist(y, bins=bins, color='#bcbddc', linewidth=0.4, orientation='horizontal')
  #add contours on top
  l = []
  if N != None:
    k = N
  else:
    k = 1
  while k < ma+1:
    l.append(k)
    k = 2 * k
  levels = np.asarray(l)
  axScatter.contour(H, levels, colors='#2171b5',linewidths=0.8)
  if chi2!=None:
    axScatter.text(25,18, "Chi^2="+str(chi2)+"$\pm$"+str(std_chi2),fontsize=8)
  if hp!=None:
    axScatter.text(25,14, "ALPHA="+str(hp)+"$\pm$"+str(std_hp),fontsize=8)
  if pp2!=None:
    axScatter.text(25,10, "PP2="+str(pp2)+"$\pm$"+str(std_pp2),fontsize=8)
  if beta!=None:
    axScatter.text(25,6, "BETA="+str(beta)+"$\pm$"+str(std_beta),fontsize=8)
  #add box onto image
  if box!=None:
  #alpha definition
    axScatter.add_patch(patches.Rectangle((12,21),18,18,fill=False,linewidth=0.8,linestyle='solid'))
  #pp2
    axScatter.add_patch(patches.Rectangle((15,60),15,9,fill=False,linewidth=0.8,linestyle='solid'))
  #beta
    axScatter.add_patch(patches.Rectangle((1,57),12,12,fill=False,linewidth=0.8,linestyle='solid'))
  plt.savefig(title+".png",dpi=300)

def subplotrama(x,y,title,N,ax=None):
  import matplotlib.pyplot as plt
  import numpy as np
  import matplotlib
  import math
  from matplotlib.ticker import NullFormatter
  from matplotlib import colors
  tick = np.arange(0, 72, 11.8)
  xedges, yedges = np.linspace(-180, 180, 72), np.linspace(-180, 180, 72)
  H,X,Y = np.histogram2d(y,x,(xedges, yedges))
  M = [max(sub_array) for sub_array in H]
  ma = int(max(M))
  im = ax.imshow(H,cmap="Purples",interpolation="None",norm=colors.Normalize(vmin=1, vmax=ma))
  ax.set_xlim((0, 72))
  ax.set_ylim((0, 72))
  l = []
  k = N
  while k < ma+1:
    l.append(k)
    k = 2 * k
  levels = np.asarray(l)
  ax.contour(H, levels, colors='#2171b5',linewidths=1)
  ax.set_xticks(tick)
  ax.set_xticklabels(['-180','-120','-60','0','60','120','180'],fontsize=10)
  ax.set_yticks(tick)
  ax.set_yticklabels(['-180','-120','-60','0','60','120','180'],fontsize=10)
  ax.set_title(title,fontsize=12,fontweight='bold')
  
def pil(name,w,h,dw,dh,outputname):
  import sys
  from PIL import Image, ImageDraw, ImageFont
  N=[]
  for i in name:
    N.append(i)
  images = map(Image.open,N)
  width, height = zip(*(i.size for i in images))
  total_width = dw*w
  total_height = dh*h
  new_im = Image.new('RGB', (total_width, total_height))
  for im in images:
    I = im.size[0]
    J = im.size[1]
  x_offset=[I*a for a in range(w)]*h
  y_offset=[]
  for a in range(h):
    y_offset.extend([J*a]*w)
  print (x_offset)
  print (y_offset)
  i=0
  for im in images:
    new_im.paste(im, (x_offset[i],y_offset[i]))
    i = i + 1

  new_im.save(outputname+".png",dpi=[300]*w*h)
