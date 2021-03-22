#!/usr/bin/env python

import argparse
from cvfit import cvfit
import cv2
import numpy as np
import os
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
    P   = np.vstack((sourceT, np.ones((1,sourceT.shape[1])))) #Stack row-wise
    Q   = np.vstack((targetT, np.ones((1,targetT.shape[1])))) #Stack row-wise
    msk = (np.min(P,axis=0) > 1/255) & (np.min(Q,axis=0) > 1/255)
    (H,err,d)   = uea_H_from_x_als(P[:,msk],Q[:,msk],10)
    #Apply 3D homography
    Pe  = H @ P #Apply chromatic homography projection
    Pe  = Pe[0:3,:]/Pe[3,:]
    Pe  = np.maximum(Pe,0) #Element-wise max - zeros-out under-saturated pixels
    Pe  = np.minimum(Pe,1) #Element-wise max - one-out over-saturated pixels
    #Brightness transfer
    PeMean = np.mean(Pe[:,msk],axis=0).T # transformed brightness
    TeMean = np.mean(targetT[:,msk],axis=0).T # target brightness
    if use_curve:
        # estimate brightness transfer
        pp = cvfit(PeMean,TeMean,'quad') # b-b mapping
    else:
        # histogram matching
        pp = cvfit(PeMean,TeMean,'hist') # b-b mapping
    #Re-apply to a higher res image
    pe      = H @ np.vstack((source.T, np.ones(1,source.T.shape[1])))
    Pe      = Pe[0:3,:]/Pe[3,:]
    Pe      = np.maximum(Pe,0) #Element-wise max - zeros-out under-saturated pixels
    Pe      = np.minimum(Pe,1) #Element-wise max - one-out over-saturated pixels
    n       = Pe.shape[1] #Number of pixels (or columns)
    PeMean  = np.mean(Pe,axis=0).T # transformed brightness
    FMean   = pp(1+np.floor(PeMean*999))
    FMean   = np.maximum(FMean,0) #Element-wise max - one-out over-saturated pixels
    D       = FMean/PeMean # convert brightness change to shading - scaling factors
    D[PeMean < (1/255)] = 1 # discard dark pixel shadings -- or scaling factor is equal to 1
    Ei      = Pe.T.reshape(source.shape)
    ImD     = D.reshape(source.shape[0:2]) #Reshape to source image size

    if use_denoise: # denoise the shading field
        grey    = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        #https://people.csail.mit.edu/sparis/bf_course/slides/03_definition_bf.pdf
        ImD     = cv.ximgproc.jointBilateralFilter(grey, ImD, d=-1, sigmaColor=0.1, sigmaSpatial=len(D)/16) #Heuristics for sigmaSpatial are 2% of length of image diagonal -- sigma color depends on mean/median of image gradients
    Ei = np.minimum(np.maximum(Ei*ImD,0),1) #Now apply shading
    return (Ei,H,pp)

def uea_H_from_x_als(P, Q, max_iter = 10, tol = 1e-20):
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

    (Nch,Npx) = P.shape[0:2]

    # definition for Graham
    fP = np.maximum(P,1e-6) 
    fQ = np.maximum(Q,1e-6)
    N = fP

    errs = [np.inf] * (max_iter+1) # error history

    # solve the homography using ALS
    n_it = 0 
    d_err = np.inf
    while ( (n_it < max_iter) and (d_err > tol) ):
        n_it    = n_it + 1 # increase number of iteration
        d       = SolveD1(N,fQ)
        P_d     = fP * d
        cv      = P_d @ P_d.T
        mma     = np.mean(np.diagonal(cv))
        cvv     = cv + (np.eye(Nch)*mma)/5000
        M       = (fQ @ P_d.T) @ np.linalg.inv(cvv)
        N       = M @ fP
        #cv      = P_d @ P_d.T
        #mma     = mean(diag(cv))
        #M       = (fQ @ P_d.T)/(cv+eye(Nch).*mma./5000)
        NDiff = (N*d-Q)**2 # squared difference
        errs[n_it] = np.mean(NDiff) # mean square error
        d_err = errs[n_it-1] - errs[n_it] # 1 order error
    H = M
    err = errs[n_it]
    return(H,err,d)

def SolveD1(p,q):
    nCh = p.shape[0]
    d = (np.ones((1,nCh)) @ (p * q)) / (np.ones((1,nCh)) @ (p * p))
    return(d)

#The default format expected is in float format
def im2double(im):
    return(skimage.img_as_float64(im))

if __name__ == "__main__":
    parsed = parse_args()
    rescale = parsed.rescale
    source = im2double(cv2.imread(parsed.source))
    target = im2double(cv2.imread(parsed.target))
    (cci, H, pp) = cf_3D_H(source, target)
    if( parsed.output ):
        cci = 255.0 * cci #Rescale back to 0-255
        cci = np.maximum(cci,0.0)
        cci = np.minimum(cci,255.0)
        cci = cci.astype('uint8')
        cv2.imwrite(parsed.output,cci)
    np.savez_compressed(parsed.homography, H=H, interpolator=pp)


