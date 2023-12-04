import os
import shutil

if not os.path.exists("./dataset"):
    print("Making dataset directory")
    os.makedirs("./dataset")

folders = ['clear', 'mixed', 'overcast']
for folder in folders:
    if not os.path.exists(f"./dataset/{folder}"):
        print(f"Making {folder} directory")
        os.makedirs(f"./dataset/{folder}")
     
    if not os.path.exists(f"./dataset/{folder}/train"):
        print(f"Making {folder}/train directory")
        os.makedirs(f"./dataset/{folder}/train")

    if not os.path.exists(f"./dataset/{folder}/test"):
        print(f"Making {folder}/test directory")
        os.makedirs(f"./dataset/{folder}/test")

    if not os.path.exists(f"./dataset/{folder}/train/HDR"):
        print(f"Making {folder}/train/HDR directory")
        os.makedirs(f"./dataset/{folder}/train/HDR")
    
    if not os.path.exists(f"./dataset/{folder}/train/LDR"):
        print(f"Making {folder}/train/LDR directory")
        os.makedirs(f"./dataset/{folder}/train/LDR")
    
    if not os.path.exists(f"./dataset/{folder}/test/LDR"):
        print(f"Making {folder}/test/LDR directory")
        os.makedirs(f"./dataset/{folder}/test/LDR")


directory = "/home/claire/Desktop/cropped_supp"
name = '_a'
directory_new = os.path.join("/home/claire/Desktop/Mice-claire-2022-08-18/labeled-data", "test_frames")

list = [fn for fn in os.listdir(directory) if name in fn]

os.mkdir(directory_new)
i = 0

for file in list:
    filename = list[i]
    src_path = os.path.join(directory, filename)
    dst_path = directory_new
    shutil.move(src_path, dst_path)
    i += 1