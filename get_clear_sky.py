import os
import shutil
import numpy as np
import sys
# create the corresponding folders
if not os.path.exists("./dataset_clear"):
    print("Making clear dataset directory")
    os.makedirs("./dataset_clear")

sub_folders = ['train', 'test']
sub_sub_folders = ['HDR', 'LDR']

for sub_folder in sub_folders:
    if not os.path.exists(f"./dataset_clear/{sub_folder}"):
        print(f"Making {sub_folder} directory")
        os.makedirs(f"./dataset_clear/{sub_folder}")
    
    for sub_sub_folder in sub_sub_folders:
        if not os.path.exists(f"./dataset_clear/{sub_folder}/{sub_sub_folder}"):
            print(f"Making {sub_folder}/{sub_sub_folder} directory")
            os.makedirs(f"./dataset_clear/{sub_folder}/{sub_sub_folder}")


test_hdr_names = [fn for fn in os.listdir("dataset/test/HDR")]
test_ldr_names = [fn for fn in os.listdir("dataset/test/LDR")]

train_hdr_names = [fn for fn in os.listdir("dataset/train/HDR")]
train_ldr_names = [fn for fn in os.listdir("dataset/train/LDR")]

count_test_hdr = 0
count_test_ldr = 0
count_train_hdr = 0
count_train_ldr = 0

for filename in test_hdr_names:
        src_path = os.path.join("dataset/test/HDR", filename)
        dst_path = "./dataset_clear/test/HDR"
        if not(os.path.isfile(dst_path)):
            shutil.copy(src_path, dst_path)
            count_test_hdr += 1
print(f"{count_test_hdr} HDR test images")

for filename in test_ldr_names:
        src_path = os.path.join("dataset/test/LDR", filename)
        dst_path = "./dataset_clear/test/LDR"
        if not(os.path.isfile(dst_path)):
            shutil.copy(src_path, dst_path)
            count_test_ldr += 1
print(f"{count_test_ldr} HDR test images")

for filename in train_hdr_names:
        src_path = os.path.join("dataset/train/HDR", filename)
        dst_path = "./dataset_clear/train/HDR"
        if not(os.path.isfile(dst_path)) and "_Cl_" in filename:
            shutil.copy(src_path, dst_path)
            count_train_hdr += 1
print(f"{count_train_hdr} HDR train images")

for filename in train_ldr_names:
        src_path = os.path.join("dataset/train/LDR", filename)
        dst_path = "./dataset_clear/train/LDR"
        if not(os.path.isfile(dst_path)) and "_Cl_" in filename:
            shutil.copy(src_path, dst_path)
            count_train_ldr += 1
print(f"{count_train_ldr} LDR train images")




