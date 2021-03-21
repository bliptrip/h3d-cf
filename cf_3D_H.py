#!/usr/bin/env python


import argparse
from cvfit import cvfit
import cv2
import functools
import numpy as np
import os
from scipy.sparse import spdiags, csr_matrix, csc_matrix, linalg, eye
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility to test functionality of 3-D color homography color transfer model algorithm based on" \
                                                                   "Gong, H., Finlayson, G.D., Fisher, R.B. and Fang, F., 2017. 3D color homography model for photo-realistic color transfer re-coding. The Visual Computer, pp.1-11.")
    parser.add_argument('-s', '--source', action='store', required=True, help="Source image filepath.")
    parser.add_argument('-t', '--target', action='store', required=True,  help="Target image filepath.")
    parser.add_argument('-H', '--homography', action='store', required=True, help="Homography matrix (chromacity + shade-mapping interpolator function) filename, stored as compressed numpy archive.")
    parser.add_argument('-r', '--rescale', type=float, default=1.0, help="Factor to scale images by before calculating homography matrix H and shading map matrix D.")
    parser.add_argument('-o', '--output', action='store', help="Color-correct image filename if specified.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

#cf_3D_H  estimates a 2-D color homography color transfer model.
#
#   CF_3D_H() returns the colour transfered source and the homography matrix
#   image source according to the target image target
#
#   Options (opt.*):
#   * downsampling_res: specify the resolution of downsampled images
#     for faster color transfer approximation. [] for disabling.
#   * use_denoise: enable deniose by bilaterial filtering.
#   * use_curve: enable tone-mapping esimation by polynomial models.
#
#   Copyright 2018 Han Gong <gong@fedoraproject.org>, University of East
#   Anglia.
#   References:
#   Gong, H., Finlayson, G.D., Fisher, R.B.: Recoding color transfer as a
#   color homography. In: British Machine Vision Conference. BMVA (2016)
def cf_3D_H(source, target, rescale=1.0, use_curve = True, use_denoise = True):
    if rescale != 1.0:
        # downsampling
        ssource = cv2.resize(source, (0,0), fx=rescale, fy=rescale).reshape((-1,3))
        starget = cv2.resize(target, (0,0), fx=rescale, fy=rescale).reshape((-1,3))
    else:
        # use full res-images
        ssource    = source.reshape((-1,3)) #Reshape to n-pixels by 3 columns for RGB
        starget    = target.reshape((-1,3)) #Reshape to n-pixes by 3 columns for RGB
    sshape     = ssource.shape
    sourceT    = ssource.T
    targetT    = starget.T
    ssshape    = sshape
    #Estimate 3D homography
    P   = np.stack([sourceT, np.ones((1,sourceT.shape[1]))], axis=1) #Stack row-wise
    Q   = np.stack([targetT, np.ones((1,targetT.shape[1]))], axis=1) #Stack row-wise
    msk = (np.min(P,axis=1) > 1/255) & (np.min(Q,axis=1) > 1/255)
    H   = uea_H_from_x_als(P[:,msk],Q[:,msk],10)
    #Apply 3D homography
    Pe  = H @ P #Apply chromatic homography projection
    Pe  = Pe[0:3,:]/Pe[3,:]
    Pe  = np.maximum(Pe,0) #Element-wise max - zeros-out under-saturated pixels
    Pe  = np.minimum(Pe,1) #Element-wise max - one-out over-saturated pixesl
    #Brightness transfer
    PeMean = mean(Pe[:,msk],axis=0).T # transformed brightness
    TeMean = mean(targetT[:,msk],axis=0).T # target brightness
    if opt.use_curve
        # estimate brightness transfer
        model.pp = cvfit(PeMean,TeMean,'quad'); # b-b mapping
    else
        # histogram matching
        model.pp = cvfit(PeMean,TeMean,'hist'); # b-b mapping
    end


    D = SolveD1(Pe,target) #Now find the difference b/w the chromacity-adjusted source and the target, to get the shading difference -- These shading values apply only within local context of identical image backdrops
    if not use_curve:
        # apply shading
        D_new   = np.minimum(D,10) # Element-wise min -- avoid shading artefact
        pp      = None
    else:
        Pem     = np.mean(Pe,axis=1)
        pp      = csfit(Pem,np.squeeze(D),50) #Cubic-spline fit - returns an interpolator that is useful tin adjusting shading at more global level
        Ped     = pp(Pem) #Calculate shade scaling factors using cubic-spline interpolator
        if use_denoise:
            D_new = SolveD2(Ped,sshape) #Adjust shading using Laplacian smoothness constraint to avoid generating shading artifacts
        else:
            D_new   = Ped
    Peo = D_new.reshape((-1,1)) * Pe #Reshape is necessary for proper broadcast
    Peo = np.reshape(Peo,sshape)
    return (Peo,H,pp)


def SolveD1(p,q):
    (nPx,nCh)       = p.shape
    d_num           = (p*q) @ np.ones((nCh,1))
    d_denom         = (p*p) @ np.ones((nCh,1))
    #d               = np.squeeze(d_num / d_denom)
    d               = d_num / d_denom
    #d[np.isnan(d)]  = 0 #Set all nan's to 0, as they mess up subsequent calculations -- These came from 0/0 division warnings
    #D               = spdiags(d.T,0,nPx,nPx)
    return(d)

def SolveD2(nd,sz):
    nPx = nd.shape[0]
    A1 = csr_matrix(eye(nPx))
    # compute D
    M1 = ShadingDiff(sz[0:2])
    mlambda = 1/(M1.diagonal().mean())
    mlambda = 0.1 * mlambda
    D = linalg.lsqr(A1+(mlambda * M1),nd)[0]
    return(D)

def ShadingDiff(lsz):
    # minimise an edge image
    lsz     = np.array(lsz)
    nel     = np.prod(lsz)
    snel    = np.prod(lsz-2)

    ind = np.array(list(range(0,nel)))
    ind = ind.reshape(lsz, order='F')
    cdx = ind[1:-1,1:-1] # centre
    tdx = ind[0:-2,1:-1] # top
    bdx = ind[2:,1:-1]   # bottom
    ldx = ind[1:-1,0:-2] # left
    rdx = ind[1:-1,2:]   # right

    # flatten index
    cdx = cdx.flatten(order='F')
    tdx = tdx.flatten(order='F')
    bdx = bdx.flatten(order='F')
    ldx = ldx.flatten(order='F')
    rdx = rdx.flatten(order='F')

    #If the images are incredibly large, then an n x n diagonal shading matrix
    #could be enormous (and consume more memory than necessary).  A sparse matrix
    #is constructed to keep memory from being chewed up excessively for a simple
    #shading mapping operation -- so we use a scipy.sparse() construct.
    sM = csr_matrix((-4*np.ones(snel),(cdx,cdx)), shape=(nel,nel)) + \
         csr_matrix((   np.ones(snel),(cdx,tdx)), shape=(nel,nel)) + \
         csr_matrix((   np.ones(snel),(cdx,bdx)), shape=(nel,nel)) + \
         csr_matrix((   np.ones(snel),(cdx,ldx)), shape=(nel,nel)) + \
         csr_matrix((   np.ones(snel),(cdx,rdx)), shape=(nel,nel))
    M2 = sM.T @ sM
    return(M2)


def uea_H_from_x_als(P, Q, max_iter, tol):
    # [H,rms,pa] = uea_H_from_x_als(H0,p1,p2,max_iter,tol)
    #
    # Compute H using alternating least square
    # An initial estimate of
    # H is required, which would usually be obtained using
    # vgg_H_from_x_linear. It is not necessary to precondition the
    # supplied points.
    #
    # The format of the xs is
    # [x1 x2 x3 ... xn  
    #  y1 y2 y3 ... yn 
    #  w1 w2 w3 ... wn]

    if nargin<3, max_iter = 10 end
    if nargin<4, tol = 1e-20 end

    [Nch,Npx] = size(P)

    # definition for Graham
    fP = max(P,1e-6) fQ = max(Q,1e-6)
    N = fP

    errs = Inf(max_iter+1,1) # error history

    # solve the homography using ALS
    n_it = 1 d_err = Inf
    while ( n_it-1<max_iter && d_err>tol)
        n_it = n_it+1 # increase number of iteration

        d = SolveD1(N,fQ)

        P_d = fP.*repmat(d,[Nch,1])
        cv=P_d*P_d' mma=mean(diag(cv))
        M = fQ*P_d'/(cv++eye(Nch).*mma./5000)
        N = M*fP

        NDiff = (N.*repmat(d,[Nch,1])-Q).^2 # difference
        errs(n_it) = mean(mean(NDiff)) # mean square error
        d_err = errs(n_it-1) - errs(n_it) # 1 order error
    end
    H = M
    err = errs(n_it)


#The default format expected is in float format
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

if __name__ == "__main__":
    parsed = parse_args()
    rescale = parsed.rescale
    source = cv2.imread(parsed.source)
    target = cv2.imread(parsed.target)
    if( rescale != 1 ):
        source = cv2.resize(source, None, fx=rescale,fy=rescale)
        target = cv2.resize(target, None, fx=rescale,fy=rescale)
    #Convert to normalized doubles as cf_3D_H expects this
    source = im2double(source)
    target = im2double(target)
    (cci, H, pp) = cf_3D_H(source, target)
    if( parsed.output ):
        cci = 255.0 * cci #Rescale back to 0-255
        cci = np.maximum(cci,0.0)
        cci = np.minimum(cci,255.0)
        cci = cci.astype('uint8')
        cv2.imwrite(parsed.output,cci)
    np.savez_compressed(parsed.homography, H=H, interpolator=pp)


