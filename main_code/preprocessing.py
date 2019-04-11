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


def create_temp_image(image_path, output_folder):
    print('creating temp image')
    temp_image_file = output_folder + '/temp_image.nii.gz'
    # using float32
    temp_image = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    sitk.WriteImage(temp_image, temp_image_file)
    return


def affine_to_atlas(atlas_wo_skull_file, args):

    print('performing affine registration')
    output_folder = args.output_folder
    # input image 
    temp_file = output_folder + '/temp_image.nii.gz'
    # affine output/trans
    affine_file = output_folder + '/affine_output.nii.gz'
    affine_trans = output_folder + '/affine_trans.txt'
    # affine trans/inv_trans
    affine_trans = output_folder + '/affine_trans.txt'
    invaff_trans = output_folder + '/affine_invtrans.txt'

    affine_log = output_folder + '/pre_affine.log'
    log = open(affine_log, 'w')
    cmd = ""
    # reg input -> atlas_w_skull, no mask
    cmd += '\n' + nifty_reg_affine(ref=atlas_wo_skull_file, flo=temp_file, res=affine_file, aff=affine_trans)
    # get inverse
    cmd += '\n' + nifty_reg_transform(invAff1=affine_trans, invAff2=invaff_trans)

    if args.verbose == True:
        print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=log)
    process.wait()
    log.close()
    return


def histogram_matching(pca_mean_file, output_folder):
    print('performing histogram matching')

    affine_file = output_folder + '/affine_output.nii.gz'
    match_file = output_folder + '/match_output.nii.gz'
    
    affine_img = sitk.ReadImage(affine_file)
    mean_img = sitk.ReadImage(pca_mean_file)

    affine_arr = sitk.GetArrayFromImage(affine_img)
    mean_arr = sitk.GetArrayFromImage(mean_img)
    img_shape = affine_arr.shape
    affine_vec = affine_arr.reshape(-1)
    mean_vec = mean_arr.reshape(-1)

    unique_b, inverse_b, counts_b = np.unique(affine_vec, return_inverse=True, return_counts=True)
    unique_m, counts_m = np.unique(mean_vec, return_counts=True)

    #match im with rm
    im_cum = np.cumsum(counts_b).astype(np.float32)
    im_qtl = im_cum/im_cum[-1]
    rm_cum = np.cumsum(counts_m).astype(np.float32)
    rm_qtl = rm_cum/rm_cum[-1]

    interp_unique_b = np.interp(im_qtl, rm_qtl, unique_m)

    match_arr = interp_unique_b[inverse_b].reshape(img_shape)
    match_img = sitk.GetImageFromArray(match_arr)
    match_img.CopyInformation(affine_img)

    sitk.WriteImage(match_img, match_file)
    del affine_img, mean_img, affine_arr, match_arr, match_img


def preprocessing(args):
    image_path = args.input_image
    image_file = os.path.basename(image_path)

    root_folder = os.path.join(sys.path[0], '..')

    if args.post_folder is not None:
        post_image = args.post_image
        post_folder = args.post_folder
        post_name = os.path.basename(post_image).split('.')[0]
        atlas_wo_skull_file = post_folder + '/temp_image_lowrank.nii.gz'
        pca_mean_file = root_folder + '/data/pca/pca_' + str(post_name) + '/mean_brain_100.nii.gz'
        
    else:
        pca_mean_file = root_folder + '/data/pca/pca_regular/mean_brain_100.nii.gz'
        atlas_wo_skull_file = root_folder + '/data/atlas/atlas_wo_skull.nii'

    create_temp_image(image_path, args.output_folder)
    affine_to_atlas(atlas_wo_skull_file, args)
    histogram_matching(pca_mean_file, args.output_folder)

if __name__ == '__main__':
    image_path = sys.argv[1]
    image_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(os.path.dirname(__file__), '../tmp_res/temp_'+image_name)
    args = Namespace(input_image=image_path, debug=2, verbose=True, output_folder=output_path)
    preprocessing(args)
