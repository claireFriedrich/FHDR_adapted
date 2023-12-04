import os
import shutil

# create right folders
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


# move the data to the right directory
data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/video_match128"
name = '_a'

liste = [fn for fn in os.listfile(data_path)]
print(liste)

i = 0

"""
for filename in liste:
    if '01_0' in filename and '_ref.hdr' in filename:
        filename = liste[i]
    src_path = os.path.join(data_path, filename)
    dst_path = directory_new
    shutil.move(src_path, dst_path)
    i += 1"""