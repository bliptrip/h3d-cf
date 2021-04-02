import numpy as np
from qpsolvers import solve_qp

def increasingF2(x, y, p = 0, a = 1000):
    #INCREASINGF2 estimates a monotonically increasing curve.
    #
    #   INCREASINGF2(x,y) returns a monotonically increasing function curve
    #   that maps x (vector of x values) to y (target vector of y values).
    #   The curve is represented in the form of LUT.
    #
    #   Options:
    #   * p: small fractional number (e.g. 0.000001)
    #     for faster color transfer approximation. [] for disabling.
    #   * a: size of LUT (e.g. 1000).
    #
    #   y ~= fout(floor(x*(a-1))+1)
    #   Copyright 2018 Graham Finlayson, Han Gong <gong@fedoraproject.org>,
    #   University of East Anglia.
    #   References:
    #   Gong, H., Finlayson, G.D., Fisher, R.B. and Fang, F., 2017. 3D color
    #   homography model for photo-realistic color transfer re-coding. The
    #   Visual Computer, pp.1-11.

    #quantise (necessary as we solve per quantisation level)
    x = np.round(x*a)/a
    #We solve for f() by quantising and histogramming
    cross = np.zeros((a+1,1))
    ticks = np.linspace(0,1,a+1).reshape((-1,1)) #Make it a column-vector
    edges = np.linspace(0-((1/a)*0.5),1+((1/a)*0.5),a+2)
    (w,xedges) = np.histogram(x, bins=edges, density=False)
    idx        = np.digitize(x, bins=xedges)
    idx        = idx - 1 #Indices are the right-side of the bins

    ticksd = ticks * np.eye(ticks.shape[0])
    cvv = ((ticks**2) * w) * np.eye(len(w)) #Convert to diagonal matrix
    for i in range(0,y.shape[0]):
        cross[idx[i]] = cross[idx[i]] + y[i]
    cross = -1 * ticks * cross

    #We use a difference function to make sure the function is increasing
    D_n = np.zeros((a,a+1))
    for i in range(0,a):
        D_n[i,i:i+2]=[-1,1]
    D_n_1 = D_n[0:-1,0:-1]

    #Now we make the 2nd derivative (note the quantised values are
    #not evenly split across the domain of x. So we take this into account
    #
    T = ((D_n_1 @ D_n) @ ticksd) * (a**2)

    # mult are the per quantisation level scalar such that z=mult*x
    sm      = T.T @ T
    P       = cvv + (sm * p)
    q       = cross.reshape((-1,))
    G       = np.vstack(((-1 * D_n) @ ticksd, ticksd))
    h       = np.vstack((np.zeros((a,1)),np.ones((a+1,1)))).reshape((-1,))
    lb      = np.zeros((a+1,))
    mult    = solve_qp(P, q, G, h, None, None, lb, None, solver="cvxopt")
    if( mult != None ):
        mult = mult.reshape((-1,1))
        fin  = np.linspace(0,1,a+1).reshape((1,-1))
        fout = fin.T * mult
    else:
        fout = None
    return(fout)
