import os

for i in range(10):
    image_file =  "/playpen/xhs400/Research/data/stripped/t1_test_data/t1_post_" + str(i+1) + ".nii.gz"
    print image_file
    cmd = "python create_patient_pca.py "+image_file
    os.system(cmd)
