import numpy as np
from scipy.interpolate import PchipInterpolator

#Cubic-spline fit function generator
def csfit(x,y,n):
    breaks          = np.linspace(start=0, stop=1, num=n+1)
    (xcount,xedges) = np.histogram(x, bins=breaks, density=False)
    xind            = np.digitize(x, bins=xedges)
    w               = xcount/np.sum(xcount)
    pointMean       = np.zeros((n,2))
    for i in range(1,n+1): #For some reason the indices returned from np.digitize() seem to start from index 1, not 0?
        pointMean[i-1,0] = np.nanmean(x[xind==i]) if (len(np.where(xind == i)[0]) > 0) else 0.0
        pointMean[i-1,1] = np.nanmean(y[xind==i]) if (len(np.where(xind == i)[0]) > 0) else 0.0
    pointMean   = pointMean[w > 0,:] #remove any blank entries
    pp          = PchipInterpolator(pointMean[:,0],pointMean[:,1])
    return(pp)
