#!/usr/bin/env python

import argparse
from csfit import csfit
import cv2
import functools
import numpy as np
import os
from scipy.sparse import spdiags, csr_matrix, csc_matrix, linalg, eye
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility to test functionality of 2-D color homography color transfer model algorithm based on" \
                                                                   "Gong, H., Finlayson, G.D., Fisher, R.B.: Recoding color transfer as a" \
                                                                   "color homography. In: British Machine Vision Conference. BMVA (2016)")
    parser.add_argument('-s', '--source', action='store', required=True, help="Source image filepath.")
    parser.add_argument('-t', '--target', action='store', required=True,  help="Target image filepath.")
    parser.add_argument('-H', '--homography', action='store', required=True, help="Homography matrix (chromacity + shade-mapping interpolator function) filename, stored as compressed numpy archive.")
    parser.add_argument('-r', '--rescale', type=float, default=1.0, help="Factor to scale images by before calculating homography matrix H and shading map matrix D.")
    parser.add_argument('-o', '--output', action='store', help="Color-correct image filename if specified.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

#cf_2D_H  estimates a 2-D color homography color transfer model.
#
#   CF_2D_H() returns the colour transfered source and the homography matrix
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
def cf_2D_H(source, target, use_curve = True, use_denoise = True):
    sshape     = source.shape
    source     = source.reshape((-1,3)) #Reshape to n-pixels by 3 columns for RGB
    target     = target.reshape((-1,3)) #Reshape to n-pixes by 3 columns for RGB
    sourceT    = source.T
    targetT    = target.T
    ssshape    = sshape
    # compute homography
    P               = source / np.sum(source, axis=1).reshape((-1,1)) #Normalize RGB values based on intensity
    P[np.isnan(P)]  = 0 #To deal with 0/0-derived nans
    Q               = target / np.sum(target, axis=1).reshape((-1,1)) #Normalize RGB values based on intensity
    Q[np.isnan(Q)]  = 0 #To deal with 0/0-derived nans
    H, err, P_d = H_from_x_als(P,Q) #Use alternate-least-squares to approximate homography matrix
    C = np.array([[1,0,1],[0,1,1],[0,0,1]], dtype=float) #Matrix to convert RGB -> RGI
    #H =  (C @ H) @ np.linalg.inv(C)
    Pe = source @ H  #Apply chromatic homography projection
    Pe = np.maximum(Pe,0) #Element-wise max - zeros-out under-saturated pixels
    #Pe = np.minimum(Pe,1) #Element-wise max - one-out over-saturated pixesl
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


def H_from_x_als(p1,p2,max_iter=20,tol=1e-20):
    (npx,nch) = p1.shape

    # definition for Graham
    A = np.maximum(p1,1e-6) #element-wise
    B = np.maximum(p2,1e-6) #element-wise

    # solve the homography using ALS
    i   = 0
    de  = np.inf
    es  = [np.inf]
    Ds  = []
    As  = []
    Hs  = []
    H   = np.eye(nch,dtype='float64')
    D   = np.ones((npx,1),dtype='float64')
    Ds.append(SolveD1(A,B)) #Least-squares estimate of D0
    D   = D * Ds[0]
    #As.append((Ds[0] @ csr_matrix(A)).toarray()) #Calculate A0
    As.append(Ds[0] * A) #Calculate A0
    while (i < max_iter) and (de > tol):
        #ACP = As[i].T @ As[i]
        #MMA = np.mean(ACP.diagonal())
        #Hs.append(np.linalg.inv(ACP + ((MMA * np.eye(nch, dtype='float64'))/1e6)) @ As[i].T @ B)
        Hs.append(np.linalg.lstsq(As[i],B)[0])
        H = H @ Hs[i]
        Ph = As[i] @ Hs[i]
        Ds.append(SolveD1(Ph, B))
        D = D * Ds[i+1]
        As.append(Ds[i+1] * Ph) 
        es.append(np.linalg.norm(As[i+1]-As[i]))
        #es.append(np.mean(Ds[i] * (A @ Hs[i])).toarray()-B)**2))
        #As.append((Ds[i+1] @ csr_matrix(A)).toarray())
        i = i + 1
        de = es[i]
        #de = np.absolute(es[i-1] - es[i]) #differential in errors
    min_i = np.argmin(es) #Which iteration had the lowest error
    #H     = Hs[min_i-1] #H-matrix is offset by 1
    e     = es[min_i]
    #D     = Ds[min_i]
    #P_d   = Pds[min_i]
    return (H,e,D)

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
    #Convert to normalized doubles as cf_2D_H expects this
    source = im2double(source)
    target = im2double(target)
    (cci, H, pp) = cf_2D_H(source, target)
    if( parsed.output ):
        cci = 255.0 * cci #Rescale back to 0-255
        cci = np.maximum(cci,0.0)
        cci = np.minimum(cci,255.0)
        cci = cci.astype('uint8')
        cv2.imwrite(parsed.output,cci)
    np.savez_compressed(parsed.homography, H=H, interpolator=pp)
