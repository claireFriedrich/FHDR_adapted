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
data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/video_match128/video_match128"
name = '_a'

list_mixed = [fn for fn in os.listdir(data_path) if fn.startswith('01')]
print("Mixed:", len(list_mixed))
list_clear = [fn for fn in os.listdir(data_path) if '_Cl' in fn]
print("Clear:", len(list_clear))
list_overcast = [fn for fn in os.listdir(data_path) if '_Ov' in fn]
print("Overcast:", len(list_overcast))


i = 0

for idx, filename in enumerate(list_mixed):
    if idx < (len(list_mixed) - 3):
        if '_ref.hdr' in filename:
            src_path = os.path.join(data_path, filename)
            dst_path = f"./dataset/mixed/train/HDR"
            if not(os.path.isfile(src_path)):
                shutil.copy(src_path, dst_path)
        if '_video.png' in filename:
            src_path = os.path.join(data_path, filename)
            dst_path = f"./dataset/mixed/train/LDR"
            if not(os.path.isfile(src_path)):
                shutil.copy(src_path, dst_path)
    else: 
        if '_video.png' in filename:
            src_path = os.path.join(data_path, filename)
            dst_path = f"./dataset/mixed/test/LDR"
            shutil.copy(src_path, dst_path)