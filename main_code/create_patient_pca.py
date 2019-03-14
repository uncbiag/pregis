import os
import sys
import SimpleITK as sitk
import subprocess
import numpy as np


sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))
from niftyreg import *
from argparse import Namespace

def create_pca(args):
    image_path = args.input_image
    image_file = os.path.basename(image_path)
    image_name = image_file.split('.')[0]

    num_of_normals = 100
    root_folder = os.path.join(sys.path[0], '..')

    pca_folder = root_folder + '/data/pca'
    oasis_folder = root_folder + '/data/oasis'

    patient_pca_folder = pca_folder + '/pca_' + image_name
    patient_oasis_folder = oasis_folder + '/oasis_' + image_name
    os.system('mkdir ' + patient_pca_folder)
    os.system('mkdir ' + patient_oasis_folder)
   
    post_folder = os.path.join(root_folder, 'tmp_res', 'post', 'tvmask_gm2', 'temp_'+image_name)
    inverse_dvf = post_folder + '/final_inv_DVF.nii'
    input_file = post_folder + '/temp_image_lowrank.nii.gz'



    # apply registration results
    cmd = ''
    for i in range(num_of_normals):
        oasis_file = oasis_folder + '/oasis/oasis_warped_' + str(i+1) + '.nii.gz'
        oasis_patient_file = patient_oasis_folder + '/patient_warped_' + str(i+1) + '.nii.gz'
        cmd += '\n' + nifty_reg_resample(ref=input_file, trans=inverse_dvf, flo=oasis_file, res=oasis_patient_file)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    # normalize image intensity
    for i in range(num_of_normals):
        print 'Normalizing image ' + str(i+1) 
        patient_file = patient_oasis_folder + '/patient_warped_' + str(i+1) + '.nii.gz'
        norm_file = patient_oasis_folder + '/patient_norm_' + str(i+1) + '.nii.gz'
        
        image = sitk.ReadImage(patient_file)
        image_arr = sitk.GetArrayFromImage(image)
        intensities = image_arr.reshape(-1)
        maxi = np.percentile(intensities,99)
        mini = np.percentile(intensities, 1)
 
        b = (maxi-99*mini)/98
        a = 0.98/(maxi-mini)

        norm_img_pre = sitk.ShiftScale(image1=image, shift=b, scale=a)

        norm_img = sitk.IntensityWindowing(norm_img_pre, windowMinimum=0.0, windowMaximum=1.0, outputMinimum=0.0, outputMaximum=1.0)

        sitk.WriteImage(norm_img, norm_file)


    # pca analysis
    im_ref = sitk.ReadImage(input_file)
    im_ref_array = sitk.GetArrayFromImage(im_ref)
    z_dim, x_dim, y_dim = im_ref_array.shape
    vector_length = z_dim * x_dim * y_dim

    D = np.zeros((vector_length,num_of_normals))

    for i in range(num_of_normals):
        print 'loading image ' + str(i+1)
        im_file2 =  patient_oasis_folder  + '/patient_norm_'+str(i+1)+'.nii.gz'
        tmp = sitk.ReadImage(im_file2)
        tmp = sitk.GetArrayFromImage(tmp)
        print np.max(tmp), np.min(tmp)
        D[:,i] = tmp.reshape(-1)

    D_mean = np.mean(D, 1, keepdims=True)
    im_meanbrain = D_mean.reshape(z_dim, x_dim,y_dim)
    img_meanbrain = sitk.GetImageFromArray(im_meanbrain)
    img_meanbrain.SetOrigin(im_ref.GetOrigin())
    img_meanbrain.SetSpacing(im_ref.GetSpacing())
    img_meanbrain.SetDirection(im_ref.GetDirection())
    fname_meanbrain = patient_pca_folder + '/mean_brain_' + str(i+1) + '.nii.gz'
    sitk.WriteImage(img_meanbrain, fname_meanbrain)



    D_centered = D - D_mean
    print D_centered.shape
    print 'starting svd'
    U, S, _ = np.linalg.svd(D_centered, full_matrices=False)

    print 'saving images'
    for i in range (num_of_normals):
        im_eigenbrain = np.array(U[:,i].reshape(z_dim, x_dim,y_dim))
        img_eigenbrain = sitk.GetImageFromArray(im_eigenbrain)
        img_eigenbrain.SetOrigin(im_ref.GetOrigin())
        img_eigenbrain.SetSpacing(im_ref.GetSpacing())
        img_eigenbrain.SetDirection(im_ref.GetDirection())
        fname_brain = patient_pca_folder + '/eigen_brain_' + str(i+1) + '.nii.gz'
        sitk.WriteImage(img_eigenbrain, fname_brain)





if __name__ == '__main__':
    image_path = sys.argv[1]
    args = Namespace(input_image=image_path, debug=2, verbose=True)
    create_pca(args)

