import os

for i in range(10):
    image_file =  "/playpen/xhs400/Research/data/stripped/t1_test_data/t1_pre_" + str(i+1) + ".nii.gz"
    post_file =  "/playpen/xhs400/Research/data/stripped/t1_test_data/t1_post_" + str(i+1) + ".nii.gz"
    print image_file
    
    cmd = "python pregis.py -i "+image_file+" -c 2 -d 2 -g 1.5 --verbose -a " + post_file
    os.system(cmd)
