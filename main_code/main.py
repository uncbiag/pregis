import sys
import numpy as np
import SimpleITK as sitk
import subprocess
import os

sys.path.append(os.path.join(sys.path[0], '..', 'func_code'))

from niftyreg import *
from decomposition import *
from operations import *
from argparse import Namespace


def performInitialization(args):
    configure = {}
    configure['num_of_normal_used'] = 100 # currently fixed number of normal images used. 2D:250;3D:80/100
    configure['num_of_pca_basis'] = 50 # currently fixed number of PCA Basis used. 2D:150;3D:50
    configure['num_of_iteration'] = 4 # currently fixed number of iteration. manually change
    configure['num_of_bspline_iteration'] = 3 # currently fixed 3
    
    configure['image_file'] = args.input_image # temp_folder + image_name
    configure['gamma'] = args.gamma # parameter, usually gamma
    configure['num_of_correction'] = args.num_of_correction # number of correction steps performed
    configure['platform'] = args.platform
    configure['start_iteration'] = 1
    configure['verbose'] = args.verbose
    configure['debug'] = args.debug

    root_folder = os.path.join(sys.path[0], '..')
    temp_folder = "temp_" + os.path.basename(configure['image_file']).split('.')[0]
    result_folder = os.path.join(root_folder, 'tmp_res', temp_folder)
    data_folder = os.path.join(root_folder, 'data')
    pca_folder = os.path.join(data_folder, 'pca')    

    after_path = args.after_image
    if after_path is not None:
        after_file = os.path.basename(after_path)
        after_name = after_file.split('.')[0]
        data_folder_basis = os.path.join(pca_folder, 'pca_' + str(after_name))
        atlas_folder = os.path.join(root_folder , 'tmp_res', 'post','tvmask_gm2','temp_' + str(after_name))
        atlas_im_name = os.path.join(atlas_folder, 'temp_image_lowrank.nii.gz')
    else:
        data_folder_basis = os.path.join(pca_folder, 'pca_regular')
        atlas_folder = os.path.join(data_folder, 'atlas')
        atlas_im_name = os.path.join(atlas_folder, 'atlas_wo_skull.nii') 

    configure['result_folder'] = result_folder
    configure['data_folder_basis'] = data_folder_basis
    configure['root_folder'] = root_folder
    configure['atlas_im_name'] = atlas_im_name
    return configure



def ReadPCABasis(image_size, configure):
    D = np.zeros((image_size, configure['num_of_pca_basis']), dtype=np.float32)
    DT = np.zeros((configure['num_of_pca_basis'], image_size), dtype=np.float32)
    if configure['verbose'] == True:
        print 'Reading PCA Basis Images'
    for i in range(configure['num_of_pca_basis']):
        basis_file = os.path.join(configure['data_folder_basis'], 'eigen_brain_'+str(i+1)+'.nii.gz')
        basis_img = sitk.ReadImage(basis_file)
        basis_img_arr = sitk.GetArrayFromImage(basis_img)
        D[:,i] = basis_img_arr.reshape(-1)
        DT[i,:] = basis_img_arr.reshape(-1)
    mean_img_file = os.path.join(configure['data_folder_basis'], 'mean_brain_'+str(configure['num_of_normal_used'])+'.nii')
    D_mean = sitk.GetArrayFromImage(sitk.ReadImage(mean_img_file)).astype(np.float32)
    return D, DT, D_mean

def createTVMask(tvIm, maskIm):
    tv_img = sitk.ReadImage(tvIm)
    mask_img_3l = sitk.OtsuMultipleThresholds(tv_img, numberOfThresholds=3)
    labelStatFilter= sitk.LabelStatisticsImageFilter()
    labelStatFilter.Execute(tv_img, mask_img_3l)
    labels = labelStatFilter.GetLabels()
    maxCount = 0
    maxLabel = labels[0]
    for i in labels:
        labelCount = labelStatFilter.GetCount(i)
        if labelCount > maxCount:
            maxLabel = i
            maxCount = labelCount 
    changeFilter = sitk.ChangeLabelImageFilter()
    changeMap = sitk.DoubleDoubleMap()
    for i in labels:
        if i == maxLabel:
            changeMap[i] = 1
        else:
            changeMap[i] = 0
    mask_img_2l = changeFilter.Execute(mask_img_3l, changeMap)
    sitk.WriteImage(mask_img_2l, maskIm)


def performIteration(configure, D_Basis, D_BasisT, D_mean, image_size):
    current_folder = configure['result_folder']
    start_iteration = configure['start_iteration']
    
    inputIm = os.path.join(configure['result_folder'], 'match_output.nii.gz')
    outputIm = os.path.join(current_folder, 'Iter1'+'_Input.nii.gz')
    
    os.system('cp ' + inputIm + ' ' + outputIm)
     
    for it in range(configure['num_of_iteration']):
        current_iter = it + 1
        if current_iter < start_iteration:
            continue
        print 'run iteration ' + str(current_iter)
        if current_iter == 1:
            # first iteration, in original space
            performDecomposition(1, current_folder, D_Basis, D_BasisT, D_mean, image_size,configure) 
        else:
            performRegistration(current_iter, current_folder, configure, registration_type = 'bspline')
            performDecomposition(current_iter, current_folder, D_Basis, D_BasisT, D_mean, image_size, configure)
            InverseToIterFirst(current_iter, current_folder, configure)   
 
    return


def InverseToIterFirst(current_iter, current_folder, configure):
    prefix = current_folder + '/Iter' + str(current_iter)
    atlas_image = configure['atlas_im_name']
    initial_input_image = os.path.join(current_folder, 'Iter1_Input.nii.gz')
    initial_input_image = current_folder + '/Iter1_Input.nii.gz'

    current_comp_def = prefix + '_def_' + str(current_iter) + '1.nii'
    current_comp_inv_def = prefix + '_inv_def_1' + str(current_iter) + '.nii'
    current_comp_inv_disp = prefix + '_inv_disp_1' + str(current_iter) + '.nii'

    lowrank_image = prefix + '_LowRank.nii.gz'
    tv_image = prefix + '_TV.nii.gz'
    inv_lowrank_image = prefix + '_InvLowRank.nii.gz'
    inv_tv_image = prefix + '_InvTV.nii.gz'
 
    cmd = ""
    cmd += '\n' + nifty_reg_transform(ref=atlas_image, invNrr1=current_comp_def, invNrr2=initial_input_image , invNrr3=current_comp_inv_def)
    cmd += '\n' + nifty_reg_transform(ref=initial_input_image, disp1=current_comp_inv_def, disp2=current_comp_inv_disp)
    cmd += '\n' + nifty_reg_resample(ref=atlas_image, flo=lowrank_image, trans = current_comp_inv_def, res=inv_lowrank_image )
    cmd += '\n' + nifty_reg_resample(ref=atlas_image, flo=tv_image, trans=current_comp_inv_def, res=inv_tv_image)
 
    logFile = open(prefix+ '_inverse' + '_data.log', 'w')
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    logFile.close()

    return 
 
 
def performRegistration(current_iter, current_folder, configure, registration_type = 'bspline'):

    atlas_image = configure['atlas_im_name']
    prefix_cur = current_folder + '/Iter' + str(current_iter) 
    prefix_pre = current_folder + '/Iter' + str(current_iter-1)

    current_input_image = prefix_cur + '_Input.nii.gz'
    initial_input_image= current_folder + '/Iter1_Input.nii.gz'
    tmp_out = current_folder + '/tmp_out.nii'
    pre_lowrank = prefix_pre + '_LowRank.nii.gz'
    current_cpp = prefix_cur + '_cpp_' + str(current_iter) + str(current_iter-1) + '.nii'
    current_def = prefix_cur + '_def_' + str(current_iter) + str(current_iter-1) + '.nii'
    pre_warped_lowrank = prefix_pre + '_warped_lowrank.nii.gz'
    if current_iter == 2:
        flo_mask = prefix_pre + '_TV.nii.gz' 
        current_comp_def = current_def
    else:
        flo_mask = False
        previous_comp_def = prefix_pre + '_def_' + str(current_iter-1) + '1.nii'
        current_comp_def = prefix_cur + '_def_' + str(current_iter) + '1.nii'
    cmd = ""
    cmd += '\n' + nifty_reg_bspline(ref=atlas_image, flo=pre_lowrank, cpp=current_cpp, res=pre_warped_lowrank, fmask=flo_mask)
    cmd += '\n' + nifty_reg_transform(ref=atlas_image, def1=current_cpp, def2=current_def)
    
    if current_iter > 2:
        cmd += '\n' + nifty_reg_transform(ref=atlas_image, ref2=atlas_image, comp1=current_def, comp2=previous_comp_def, comp3=current_comp_def) # composite

    cmd += '\n' + nifty_reg_resample(ref=atlas_image, flo=initial_input_image, trans=current_comp_def, res=current_input_image)

    print 'performing registration'
    if configure['verbose'] == True:
        print cmd
    logFile = open(prefix_cur + '_data.log', 'w')
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    logFile.close()
    return 
             

        
def performDecomposition(current_iter, current_folder, Beta, BetaT, D_mean, image_size, configure):

    num_of_iteration = configure['num_of_iteration']
    num_of_bspline_iter = configure['num_of_bspline_iteration']

    _gamma = configure['gamma']
    atlas_im_name = configure['atlas_im_name']
    correction = configure['num_of_correction']
    
    input_name = current_folder + '/' + 'Iter' + str(current_iter) + '_Input.nii.gz'
    D = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).astype(np.float32)
    if current_iter <= 2 and correction != 0:
        if configure['platform'] == 'CPU':
            L, T, Alpha = pca_CPU(D, D_mean, Beta, _gamma , 0, configure['verbose'])
        else:
            L, T, Alpha = pca_GPU(D, D_mean, Beta, BetaT, _gamma, 0, configure['verbose'])
    else:
        if configure['platform'] == 'CPU':
            L, T, Alpha = pca_CPU(D, D_mean, Beta, _gamma, correction, configure['verbose'])
        else:
            L, T, Alpha = pca_GPU(D, D_mean, Beta, BetaT, _gamma, correction, configure['verbose'])
 

    l_v = L.reshape(image_size, 1) # quasi low-rank/reconstruction
    t_v = T.reshape(image_size, 1) # total variation term
    prefix = current_folder + '/Iter' + str(current_iter)
 
    lowRankIm = prefix + '_LowRank.nii.gz'
    totalIm = prefix + '_TV.nii.gz'
    masktvIm = prefix+ '_TVmask.nii.gz'
    save_image_from_data_matrix(l_v,atlas_im_name,lowRankIm)
    save_image_from_data_matrix(t_v,atlas_im_name,totalIm)
    createTVMask(totalIm, masktvIm)
       

    return


def main(args):
    configure = performInitialization(args)
    atlas_arr = sitk.GetArrayFromImage(sitk.ReadImage(configure['atlas_im_name']))
    z,x,y = atlas_arr.shape
    image_size = x*y*z

    D_Basis, D_BasisT, D_mean = ReadPCABasis(image_size, configure)
    performIteration(configure, D_Basis, D_BasisT, D_mean, image_size)

if __name__ == '__main__':
    args = Namespace(input_image=sys.argv[1], gamma=float(sys.argv[3]), num_of_correction=int(sys.argv[4]), platform=sys.argv[5], debug=1, verbose=True)
    main(args)
 

