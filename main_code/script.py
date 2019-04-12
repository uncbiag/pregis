import subprocess

pre_image = '/playpen/xhs400/Research/PycharmProjects/pregis/data/test_image/t1_pre_1.nii.gz'
post_image = '/playpen/xhs400/Research/PycharmProjects/pregis/data/test_image/t1_post_1.nii.gz'

pre_folder = '/playpen/xhs400/Research/PycharmProjects/pregis/tmp_res/test_pre'
post_folder = '/playpen/xhs400/Research/PycharmProjects/pregis/tmp_res/test_post'

cmd = ""
cmd += '\n' + "python pregis.py -i " + post_image + " -o " + post_folder +  " -c 2 -g 2"
cmd += '\n' + "python create_patient_pca.py " + post_image + " " + post_folder
cmd += '\n' + "python pregis.py -i " + pre_image + " -o " + pre_folder + " -pi " + post_image + " -pf "+ post_folder  + " -c 2 -g 2"

process = subprocess.Popen(cmd, shell=True)
process.wait()
