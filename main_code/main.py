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
import glob

def performInitialization(args):
    configure = {}
    configure['num_of_normal_used'] = 100 # currently fixed number of normal images used. 2D:250;3D:80/100
    configure['num_of_pca_basis'] = 50 # currently fixed number of PCA Basis used. 2D:150;3D:50
    configure['num_of_iteration'] = 6 # currently fixed number of iteration. manually change
    configure['num_of_bspline_iteration'] = 5 # currently fixed 5
    
    configure['image_file'] = args.input_image # temp_folder + image_name
    configure['gamma'] = args.gamma # parameter, usually gamma
    configure['num_of_correction'] = args.num_of_correction # number of correction steps performed
    configure['platform'] = args.platform
    configure['start_iteration'] = 1
    configure['verbose'] = args.verbose
    configure['debug'] = args.debug

    root_folder = os.path.join(sys.path[0], '..')
    result_folder = args.output_folder
    
    data_folder = os.path.join(root_folder, 'data')
    pca_folder = os.path.join(data_folder, 'pca')    

    if args.post_folder is not None:
        post_image = args.post_image
        post_folder = args.post_folder
        post_file = os.path.basename(post_image)
        post_name = post_file.split('.')[0]
        data_folder_basis = pca_folder + '/pca_' + str(post_name)
        atlas_im_name = post_folder + '/temp_image_lowrank.nii.gz'
    else:
        data_folder_basis = pca_folder + '/pca_regular'
        atlas_folder = os.path.join(data_folder, 'atlas')
 
        atlas_im_name = atlas_folder + '/atlas_wo_skull.nii'

    configure['result_folder'] = result_folder
    configure['data_folder_basis'] = data_folder_basis
    configure['root_folder'] = root_folder
    configure['atlas_im_name'] = atlas_im_name
    return configure



def ReadPCABasis(image_size, configure):
    D = np.zeros((image_size, configure['num_of_pca_basis']), dtype=np.float32)
    DT = np.zeros((configure['num_of_pca_basis'], image_size), dtype=np.float32)
    if configure['verbose'] == True:
        print('Reading PCA Basis Images')
    for i in range(configure['num_of_pca_basis']):
        basis_file = configure['data_folder_basis'] + '/eigen_brain_' + str(i+1) + '.nii.gz'
        basis_img = sitk.ReadImage(basis_file)
        basis_img_arr = sitk.GetArrayFromImage(basis_img)
        D[:,i] = basis_img_arr.reshape(-1)
        DT[i,:] = basis_img_arr.reshape(-1)
    mean_img_file = configure['data_folder_basis'] + '/mean_brain_'+str(configure['num_of_normal_used'])+'.nii.gz'
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
    
    inputIm = configure['result_folder'] + '/match_output.nii.gz'
    outputIm = current_folder + '/Iter1' + '_Input.nii.gz'
    
    os.system('cp ' + inputIm + ' ' + outputIm)
     
    for it in range(configure['num_of_iteration']):
        current_iter = it + 1
        if current_iter < start_iteration:
            continue
        print('run iteration ' + str(current_iter))
        if current_iter == 1:
            # first iteration, in original space
            performDecomposition(1, current_folder, D_Basis, D_BasisT, D_mean, image_size,configure) 
        else:
            if configure['num_of_iteration'] - current_iter > configure['num_of_bspline_iteration'] - 1:
                # only perform affine registration
                performRegistration(current_iter, current_folder, configure, registration_type = 'affine')
                performDecomposition(current_iter, current_folder, D_Basis, D_BasisT, D_mean, image_size, configure)
            else:
                performRegistration(current_iter, current_folder, configure, registration_type = 'bspline')
                performDecomposition(current_iter, current_folder, D_Basis, D_BasisT, D_mean, image_size, configure)
                #performInverse(current_iter, current_folder, configure)
            InverseToIterFirst(current_iter, current_folder, configure)   
    createCompDisp(current_folder, configure) 
    if configure['debug'] != 2:
        clearUncessaryFiles(current_folder, configure)
 
    return

def createCompDisp(current_folder, configure):
    atlas_im_name = configure['atlas_im_name']
    
    current_comp_disp = current_folder + '/Iter6_DVF_61.nii'
    temp_image = current_folder + '/temp_image.nii.gz'
    affine_txt = current_folder + '/affine_trans.txt'
    affine_def = current_folder + '/affine_DEF.nii'
    affine_disp = current_folder + '/affine_DVF.nii'
    final_disp = current_folder + '/final_DVF.nii'
    final_inv_disp = current_folder + '/final_inv_DVF.nii'

    final_lowrank_img = current_folder + '/temp_image_lowrank.nii.gz' 
    final_iter_lowrank = current_folder + '/Iter'+str(configure['num_of_iteration']) + '_LowRank.nii.gz'

    cmd = ""
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,def1=affine_txt, def2=affine_def)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=affine_def, disp2=affine_disp)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, ref2=atlas_im_name, comp1=current_comp_disp, comp2=affine_disp, comp3=final_disp)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=final_disp, invNrr2=temp_image, invNrr3=final_inv_disp)
    cmd += '\n' + nifty_reg_resample(ref=temp_image, flo=final_iter_lowrank, trans=final_inv_disp, res=final_lowrank_img)

    logFile = open(current_folder + '/final.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()



def clearUncessaryFiles(current_folder, configure):
    final_inv_disp = 'final_inv_DVF.nii'
    final_lowrank = 'temp_image_lowrank.nii.gz'
    files_to_keep = [final_inv_disp, final_lowrank]
    files = glob.glob(os.path.join(current_folder, '*'))
    for f in files:
        f_name = os.path.basename(f)
        if f_name in files_to_keep:
            continue
        #print("Clear File {}".format(f_name))
        os.system('rm '+f) 
 



def performInverse(current_iter, current_folder, configure):
    prefix = current_folder + '/Iter' + str(current_iter)
    atlas_im_name = configure['atlas_im_name']
    invWarpedLowRankIm = prefix + '_InvWarpedLowRank.nii.gz'
    lowRankIm = prefix + '_LowRank.nii.gz'
 
    tvIm = prefix + '_TV.nii.gz'
    invTV 
    
    cmd = ""
    current_disp = prefix + '_DVF_'+str(current_iter)+'1.nii'
    current_inv_disp =  prefix + '_InvDVF_' + str(current_iter) + '1.nii'
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=current_disp, invNrr2=lowRankIm , invNrr3=current_inv_disp)
    cmd += '\n' + nifty_reg_resample(ref=atlas_im_name, flo=lowRankIm, trans=current_inv_disp, res=invWarpedLowRankIm)
       
    if configure['verbose'] == True:
        print('Non-greedy strategy, inversing to original space')
        print(cmd)
    logFile = open(prefix + '_data2.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()


def InverseToIterFirst(current_iter, current_folder, configure):
    prefix = current_folder + '/Iter' + str(current_iter)
    inverse_disp = prefix + '_inverseDVF_1'+str(current_iter) + '.nii'

    lowRankIm = prefix + '_LowRank.nii.gz'
    totalIm = prefix + '_TV.nii.gz'
    
    invLowRankIm = prefix + '_InvWarpedLowRank.nii.gz'
    invTotalIm = prefix + '_InvTV.nii.gz'
    invTVMaskIm = prefix + '_InvTVMask.nii.gz'
    
    cmd = ""
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=lowRankIm, trans = inverse_disp, res=invLowRankIm)
    cmd += '\n' + nifty_reg_resample(ref=configure['atlas_im_name'], flo=totalIm, trans = inverse_disp, res=invTotalIm)
 
    logFile = open(prefix+ '_inverse_final_image' + '_data.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
    process.wait()
    logFile.close()
    createTVMask(invTotalIm, invTVMaskIm)

    return 
 
 
def performRegistration(current_iter, current_folder, configure, registration_type = 'bspline'):
    atlas_im_name = configure['atlas_im_name']
    prefix = current_folder + '/' + 'Iter' + str(current_iter) 
    new_input_image = prefix + '_Input.nii.gz'
    current_input_image = current_folder + '/Iter' + str(current_iter-1) + '_Input.nii.gz'
    initial_input_image= current_folder+'/Iter1_Input' + '.nii.gz'
    tmp_out = current_folder + '/tmp_out.nii'
    
    fmaskIm = False
    if configure['num_of_iteration'] - current_iter > configure['num_of_bspline_iteration'] - 2:
        movingIm = current_folder + '/Iter' + str(current_iter-1) + '_LowRank.nii.gz'
        fmaskIm = current_folder + '/Iter' + str(current_iter-1) + '_TVmask.nii.gz'
        current_def = prefix + '_DEF_'+str(current_iter)+str(current_iter-1)+'.nii'
        current_disp = prefix + '_DVF_'+str(current_iter)+str(current_iter-1)+'.nii'
    else:
        movingIm = current_folder + '/Iter' + str(current_iter-1) + '_InvWarpedLowRank.nii.gz'

        #fmaskIm = current_folder + '/Iter' + str(current_iter-1) + '_InvTVMask.nii.gz'

        current_def = prefix + '_DEF_'+str(current_iter)+'1.nii'
        current_disp = prefix + '_DVF_'+str(current_iter)+'1.nii'
    cmd  = ""
    if registration_type == 'affine':
        outputTransform = prefix + '_Transform.txt' 
        cmd += '\n' + nifty_reg_affine(ref=atlas_im_name, flo=movingIm, aff=outputTransform, symmetric=False, res=tmp_out, fmask=fmaskIm)
    else:
        outputTransform = prefix + '_Transform.nii'
        cmd += '\n' + nifty_reg_bspline(ref=atlas_im_name, flo=movingIm, cpp=outputTransform, res=tmp_out, fmask = fmaskIm)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,def1=outputTransform, def2=current_def)
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=current_def, disp2=current_disp)


    current_comp_def = current_def
    current_comp_disp = current_disp

    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name,disp1=current_comp_def,disp2=current_comp_disp)

    inverse_disp = prefix + '_inverseDVF_1'+str(current_iter)+'.nii'
    cmd += '\n' + nifty_reg_transform(ref=atlas_im_name, invNrr1=current_comp_disp, invNrr2=initial_input_image, invNrr3=inverse_disp)
    cmd += '\n' + nifty_reg_resample(ref=atlas_im_name,flo=initial_input_image,trans=current_comp_def,res=new_input_image)
    print('performing registration')
    if configure['verbose'] == True:
        print(cmd)
    logFile = open(prefix + '_data.log', 'w')
    process = subprocess.Popen(cmd, stdout= logFile, shell = True)
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
    if current_iter <= num_of_iteration-3 and correction != 0:
        if configure['platform'] == 'CPU':
            L, T, Alpha = pca_CPU(D, D_mean, Beta, _gamma/2 , 0, configure['verbose'])
        else:
            L, T, Alpha = pca_GPU(D, D_mean, Beta, BetaT, _gamma/2, 0, configure['verbose'])
    else:
        if configure['platform'] == 'CPU':
            L, T, Alpha = pca_CPU(D, D_mean, Beta, _gamma, correction, configure['verbose'])
        else:
            L, T, Alpha = pca_GPU(D, D_mean, Beta, BetaT, _gamma, correction, configure['verbose'])
 

    l_v = L.reshape(image_size, 1) # quasi low-rank/reconstruction
    t_v = T.reshape(image_size, 1) # total variation term
    prefix = current_folder + '/' + 'Iter' + str(current_iter)
 
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
 

