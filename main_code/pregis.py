import argparse
import os
import warnings

from preprocessing import *
from main import *
from create_patient_pca import create_pca
support_extensions = ['img', 'hdr', 'nii', 'nii.gz', 'gz', 'head', 'brik', 'ima', 'gipl', 'mha', 'nrrd']


def is_folder_exist(folder_fullpath):
    if not os.path.isdir(folder_fullpath):
        print('The output folder does not exist! Creating output folder')
        os.system('mkdir -p {}'.format(folder_fullpath))
        if not os.path.isdir(folder_fullpath):
            raise argparse.ArgumentTypeError("The output folder can not be created!")
    return folder_fullpath


def is_file_exist(file_fullpath):
    if not os.path.isfile(file_fullpath):
        msg = 'The input image file does not exist!'
        raise argparse.ArgumentTypeError(msg)

#    base_name = os.path.basename(file_fullpath)
#    extension = '.'.join(base_name.split('.')[1:])
#   check file extension  (FUTURE)
    #if not extension in support_extensions:
        #msg = 'The input image file extension is not supported!'
        #raise argparse.ArgumentTypeError(msg)
    return file_fullpath

def pathological_registration(args):
    print('Starting pre-processing')
    preprocessing(args)
    print('Starting Decomposition/Registration')
    main(args)
    return 


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='PCA model for Pathological Image Registration: We provide arguments for user to specify, although the default settings are sufficient for most of the cases.', prog='PRegis', conflict_handler='resolve', usage='python pregis.py -i Input_Image -o Output_Folder [-p platform] [-g gamma] [-c correction] [-v] [-h]')

    required_args = parser.add_argument_group(title='required arguments (One and only one of (-f and -i) is required)')
    required_args.add_argument('-i', '--input', required=True, metavar='', dest='input_image', help='input image name', type=is_file_exist)
    required_args.add_argument('-o', '--output', metavar='', dest='output_folder', help='output folder', type=is_folder_exist)

    additional_args = parser.add_argument_group('additional arguments')
    additional_args.add_argument('-p', '--platform', metavar='', dest='platform', help='platform (CPU/GPU)', default='GPU', choices=['CPU', 'GPU', 'cpu', 'gpu'])
    additional_args.add_argument('-pf', '--post-folder', metavar='', dest='post_folder', help='post operation image result folder', type=is_folder_exist)
    additional_args.add_argument('-pi', '--post-image', metavar='', dest='post_image', help='post operation image', type=is_file_exist)
    additional_args.add_argument('-g', '--gamma', metavar='', dest='gamma', help='gamma for total variation term penalty, default 0.5', type=float, default=0.5)
    additional_args.add_argument('-c', '--correction', metavar='', dest='num_of_correction', help='number of correction (regularization) steps, default 0', type=int, default=0)
    additional_args.add_argument('-d', '--debug', metavar='', dest='debug', help='Debug mode:\n 0: no intermediate results will be saved. \n 1: some of itermediate results will be saved. \n 2: all intermediate results will be saved in result folder. Be careful, this could occupy large space on disk if multiply images will be processed', type=int, choices=[0,1,2], default=0)
    additional_args.add_argument('-v', '--version', action='version', version='PRegis v1.0: PCA model for Pathological image registration')
    additional_args.add_argument('-h', '--help', action='help', help='show this help message and exit')
    additional_args.add_argument('-V', '--verbose', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    
    if args.debug == 2:
        msg = 'You are using ultra debugging mode. All intermediate results will be saved on disk. This could occupy large space on disk if multiple images will be processed. Consider use -d 1.'
        warnings.warn(message=msg, category=Warning)

    args.platform = args.platform.upper()

    if args.post_folder is not None:
        assert(args.post_image is not None)
    if args.post_image is not None:
        assert(args.post_folder is not None)

    if args.output_folder is None:
        tmp_folder = os.path.join(os.path.dirname(__file__), '../tmp_res')
        image_name = os.path.basename(args.input_image).split('.')[0]

        output_folder = os.path.join(tmp_folder, 'temp_'+image_name)
        os.system('mkdir -p ' + output_folder)
        args.output_folder = output_folder
    print("Program Starting")
    print(args)
    pathological_registration(args)
