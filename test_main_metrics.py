#!/usr/bin/env python
#

import argparse
import cv2
import numpy as np
import os
import pandas as pd
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
    parser.add_argument('-a', '--color_approximation_methods', action='append', default=['2D_H'], help="List of new approximation methods applied to derive images in 'out' folder.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == "__main__":
    parsed = parse_args()
    # configuration
    in_path = parsed.in_path # enchancement output path
    out_path = parsed.out_path # output path

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
        for j,cf in enumerate(cf_names):
            f_out   = '{}/{}_{}.jpg'.format(in_path,i,cf)
            target_uint8 = cv2.imread(f_out)
            for k,ap in enumerate(ap_names):
                f_ap = '{}/{}_{}_{}.jpg'.format(out_path,i,cf,ap)
                source_uint8 = cv2.imread(f_ap)
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
