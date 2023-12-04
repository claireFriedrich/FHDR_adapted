"""File to split the downloaded data (tile) into train and test"""

import os
import shutil


# ========================================
# Folder creating
# ========================================

# create dataset folder
if not os.path.exists("./dataset"):
    print("Making dataset directory")
    os.makedirs("./dataset")

# create train folder
if not os.path.exists(f"./dataset/train"):
        print(f"Making train directory")
        os.makedirs(f"./dataset/train")

# create test folder
if not os.path.exists(f"./dataset/test"):
        print(f"Making test directory")
        os.makedirs(f"./dataset/test")

# create HDR train folder
if not os.path.exists(f"dataset/train/HDR"):
    print(f"Making train HDR directory")
    os.makedirs(f"dataset/train/HDR")

# create LDR train folder 
if not os.path.exists(f"dataset/train/LDR"):
    print(f"Making train LDR directory")
    os.makedirs(f"dataset/train/LDR")

# create HDR test folder
if not os.path.exists(f"dataset/test/HDR"):
    print(f"Making test HDR directory")
    os.makedirs(f"dataset/test/HDR")
    
# create LDR test folder
if not os.path.exists(f"dataset/test/LDR"):
    print(f"Making test LDR directory")
    os.makedirs(f"dataset/test/LDR")


# ========================================
# Moving the training data into the corresponding folders
# ========================================

data_path_train = "C:/Users/Céline Kalbermatten/Documents/EPFL/MA1/Machine_Learning/video_match_new/tile"

# ref -> HDR ground truth image
list_ref_train = [fn for fn in os.listdir(data_path_train) if '_ref' in fn]
print("HDR ground truth:", len(list_ref_train))

# video -> LDR image
list_video_train = [fn for fn in os.listdir(data_path_train) if '_video' in fn]
print("Ref:", len(list_video_train))

lists_training = [list_ref_train, list_video_train]

for i in range(len(lists_training)):
    for filename in lists_training[i]:
        if '_ref' in filename:
            src_path = os.path.join(data_path_train, filename)
            dst_path = f"./dataset/train/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
        
        if '_video' in filename:
            src_path = os.path.join(data_path_train, filename)
            dst_path = f"./dataset/train/LDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)

    
# ========================================
# Moving the test data into the corresponding folders
# ========================================

data_path_test = "C:/Users/Céline Kalbermatten/Documents/EPFL/MA1/Machine_Learning/video_match_new"

# ref -> HDR ground truth image
list_ref_test = [fn for fn in os.listdir(data_path_test) if '_ref' in fn]
print("HDR ground truth:", len(list_ref_test))

# video -> LDR image
list_video_test = [fn for fn in os.listdir(data_path_test) if '_video' in fn]
print("Ref:", len(list_video_test))

lists_test = [list_ref_test, list_video_test]

for i in range(len(lists_test)):
    for filename in lists_test[i]:
        if '_ref' in filename:
            src_path = os.path.join(data_path_test, filename)
            dst_path = f"./dataset/test/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
        
        if '_video' in filename:
            src_path = os.path.join(data_path_test, filename)
            dst_path = f"./dataset/test/LDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
    

print('All files have been moved accordingly')
