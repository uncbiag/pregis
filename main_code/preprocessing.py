import sys
import subprocess
import os
import SimpleITK as sitk
import numpy as np
import warnings

sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))
from niftyreg import *
from operations import *
from argparse import Namespace

def create_temp_image(image_path, temp_folder):
    print 'creating temp image'
    temp_image_file = temp_folder + '/temp_image.nii.gz'
    # using float32
    temp_image = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    sitk.WriteImage(temp_image, temp_image_file)
    del temp_image 

def affine_to_atlas(atlas_wo_skull_file, temp_folder, args):

    print 'performing affine registration'
    # input image 
    temp_file = temp_folder + '/temp_image.nii.gz'
    # affine output/trans
    affine_file = temp_folder + '/affine_output.nii.gz'
    affine_trans = temp_folder + '/affine_trans.txt'
    # affine trans/inv_trans
    affine_trans = temp_folder + '/affine_trans.txt'
    invaff_trans = temp_folder + '/affine_invtrans.txt'

    affine_log = temp_folder + '/pre_affine.log'
    log = open(affine_log, 'w')
    cmd = ""
    # reg input -> atlas_w_skull, no mask
    cmd += '\n' + nifty_reg_affine(ref=atlas_wo_skull_file, flo=temp_file, res=affine_file, aff=affine_trans, symmetric=False, init = 'cog')
    # get inverse
    cmd += '\n' + nifty_reg_transform(invAff1=affine_trans, invAff2=invaff_trans)

    if args.verbose == True:
        print cmd
    process = subprocess.Popen(cmd, shell=True, stdout=log)
    process.wait()
    log.close()

    
    #remove_nan(affine_file) 


def bias_correction(atlas_erode_mask_file, temp_folder):
    print 'performing bias correction' 

    # input/output file name
    affine_file = temp_folder + '/affine_output.nii.gz'
    bias_file = temp_folder + '/bias_output.nii.gz'
     
    # input image
    affine_img = sitk.ReadImage(affine_file)
    
    mask_img = sitk.ReadImage(atlas_erode_mask_file)
    bias_img = sitk.N4BiasFieldCorrection(image=affine_img, maskImage=mask_img)

    sitk.WriteImage(bias_img, bias_file)
    del affine_img, bias_img

def intensity_normalization(temp_folder):
    print 'performing intensity normalization'
    
    # input/output file name
    affine_file = temp_folder + '/affine_output.nii.gz'
    norm_file = temp_folder + '/norm_output.nii.gz'
    
    affine_img = sitk.ReadImage(affine_file)
    affine_arr = sitk.GetArrayFromImage(affine_img)
    
    # calculate 99th and 1st percentile
    intensities = affine_arr.reshape(-1)
    i_max = np.percentile(intensities, 99)
    i_min = np.percentile(intensities, 1)
 
    # map i_max -> 900, i_min -> 100, affine tranform on intensities, then cutoff [0, 1]
    # y = a(x+b)
    b = (i_max-9*i_min)/8
    a = 0.8/(i_max-i_min)
    
    norm_img_pre = sitk.ShiftScale(image1=affine_img, shift=b, scale=a)

    # cutoff at 0, 1000
    norm_img = sitk.IntensityWindowing(norm_img_pre, windowMinimum=0.0, windowMaximum=1.0, outputMinimum=0.0, outputMaximum=1.0)

    sitk.WriteImage(norm_img, norm_file)
    del affine_img, norm_img_pre, norm_img
   
def histogram_matching(pca_mean_file, temp_folder):
    print 'performing histogram matching'

    bias_file = temp_folder + '/temp_image.nii.gz'
    match_file = temp_folder + '/match_output.nii.gz'
    
    bias_img = sitk.ReadImage(bias_file)
    mean_img = sitk.ReadImage(pca_mean_file)

    bias_arr = sitk.GetArrayFromImage(bias_img)
    mean_arr = sitk.GetArrayFromImage(mean_img)
    img_shape = bias_arr.shape
    bias_vec = bias_arr.reshape(-1)
    mean_vec = mean_arr.reshape(-1)

    unique_b, inverse_b, counts_b = np.unique(bias_vec, return_inverse=True, return_counts=True)
    unique_m, counts_m = np.unique(mean_vec, return_counts=True)

    #match im with rm
    im_cum = np.cumsum(counts_b).astype(np.float32)
    im_qtl = im_cum/im_cum[-1]
    rm_cum = np.cumsum(counts_m).astype(np.float32)
    rm_qtl = rm_cum/rm_cum[-1]

    interp_unique_b = np.interp(im_qtl, rm_qtl, unique_m)

    match_arr = interp_unique_b[inverse_b].reshape(img_shape)
    match_img = sitk.GetImageFromArray(match_arr)
    match_img.CopyInformation(bias_img)

    sitk.WriteImage(match_img, match_file)
    del bias_img, mean_img, bias_arr, match_arr, match_img


def preprocessing(args):
    image_path = args.input_image
    image_file = os.path.basename(image_path)
    image_name = image_file.split('.')[0] 

    root_folder = os.path.join(sys.path[0], '..')

    after_path = args.after_image
    if after_path is not None:
        after_file = os.path.basename(after_path)
        after_name = after_file.split('.')[0]
        atlas_wo_skull_file = root_folder + '/tmp_res/post/tvmask_gm2/temp_' + str(after_name) + '/temp_image_lowrank.nii.gz'
        pca_mean_file = root_folder + '/data/pca/pca_' + str(after_name) + '/mean_brain_100.nii.gz'
        
    else:
        pca_mean_file = root_folder + '/data/pca/pca_regular/mean_brain_100.nii.gz'
        atlas_wo_skull_file = root_folder + '/data/atlas/atlas_wo_skull.nii'

    temp_folder = os.path.join(os.sys.path[0], '..', 'tmp_res', 'temp_'+image_name)
    if os.path.exists(temp_folder):
        msg = 'The temp folder exists. You may have tried to extract the brain from this image. The previous results will be overwrited!'
        warnings.warn(message = msg, category=Warning) 
    else:    
        os.system('mkdir ' + temp_folder)

    create_temp_image(image_path, temp_folder)
 
    affine_to_atlas(atlas_wo_skull_file, temp_folder, args)
    histogram_matching(pca_mean_file, temp_folder) 
    intensity_normalization(temp_folder)

if __name__ == '__main__':
    image_path = sys.argv[1]
    args = Namespace(input_image=image_path, debug=2, verbose=True)
    preprocessing(args)
