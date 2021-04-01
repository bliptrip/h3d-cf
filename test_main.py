#!/usr/bin/env python
#
#   'test_main.m' is the main evaluation script for testing vairous color
#   transfer approximation methods. The error between the original color
#   transfer output and an approximated one is measured in PSNR an SSIM.
#   Also, we show a one-way anova post hoc test to analyse the similarity
#   between different results.

#   Copyright 2018 Han Gong <gong@fedoraproject.org>, University of East
#   Anglia.

#   References:
#   Gong, H., Finlayson, G.D., Fisher, R.B. and Fang, F., 2017. 3D color
#   homography model for photo-realistic color transfer re-coding. The
#   Visual Computer, pp.1-11.

import argparse
import cv2
import numpy as np
import os
import pandas as pd
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Loop through a set of folders and calculate the total and mean PSNR and SSIM metrics\
                                                                    for differences b/w images.")
    parser.add_argument('-i', '--in', dest="in_path", action='store', type=str, default="./in_pair", help="Directory with source images and original color transfer targets.")
    parser.add_argument('-o', '--out', dest="out_path", action='store', type=str, default="./out_pair", help="Directory with new color transfer method images or each source/transfer target pair.")
    parser.add_argument('-s', '--start', action='store', type=int, default=1, help="Starting image index.")
    parser.add_argument('-e', '--end', action='store', type=int, default=200, help="Ending image index.")
    parser.add_argument('-c', '--color_transfer_methods', action='append', default=['nguyen', 'pitie','pouli11','reinhard'], help="List of original color transfer methods applied to source images in 'in' folder.")
    parser.add_argument('-a', '--color_approximation_methods', action='append', default=['3D_H'], help="List of new approximation methods applied to derive images in 'out' folder.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

def im2float(im):
    return(skimage.img_as_float32(im))

def im2double(im):
    return(skimage.img_as_float64(im))

def im2u8(im):
    return(skimage.img_as_ubyte(im))

if __name__ == "__main__":
    parsed = parse_args()
    # configuration
    in_path = parsed.in_path # enchancement output path
    out_path = parsed.out_path # output path

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # define colour enhancement methods
    cf_names = parsed.color_transfer_methods
    ap_names = parsed.color_approximation_methods
    Ncf = len(cf_names) # number of color transfer methods
    Nap = len(ap_names) # number of approximation methods

    # discover images
    Nf = (parsed.end - parsed.start) + 1

    # Tables to compare color approximation methods: PSNR and SSIM
    a_psnr = np.zeros((Ncf,Nap,Nf), dtype='float64')
    a_psnr[:] = np.nan #initialize with nan's
    a_ssim = np.zeros((Ncf,Nap,Nf), dtype='float64')
    a_ssim[:] = np.nan #initialize with nan's

    for i in range(parsed.start,parsed.end+1):
        source    = im2double(cv2.imread('{}/{}_s.jpg'.format(in_path,i))) # input image i
        for j,cf in enumerate(cf_names):
            f_out   = '{}/{}_{}.jpg'.format(in_path,i,cf)
            target_uint8 = cv2.imread(f_out)
            target  = im2double(target_uint8)
            for k,ap in enumerate(ap_names):
                ap_import = "from cf_{} import cf_{}".format(ap,ap)
                exec(ap_import)
                ap_fun = "cf_{}(source,target,use_curve=True,use_denoise=False,rescale=0.125)".format(ap)
                (source_transformed, H, pp) = eval(ap_fun)
                f_ap = '{}/{}_{}_{}.jpg'.format(out_path,i,cf,ap)
                source_uint8 = (source_transformed*255.0).astype('uint8')
                cv2.imwrite(f_ap, source_uint8)
                a_psnr[j,k,i-1] = peak_signal_noise_ratio(source_uint8, target_uint8)
                a_ssim[j,k,i-1] = structural_similarity(source_uint8,target_uint8,multichannel=True)
    # get mean
    m_psnr = np.mean(a_psnr, axis=2)
    m_ssim = np.mean(a_ssim, axis=2)
    # construct tables
    T_psnr = pd.DataFrame(m_psnr.T, columns=cf_names, index=ap_names)
    T_psnr.to_csv("{}/psnr_means.csv".format(out_path))
    T_ssim = pd.DataFrame(m_ssim.T, columns=cf_names, index=ap_names)
    T_ssim.to_csv("{}/ssim_means.csv".format(out_path))
