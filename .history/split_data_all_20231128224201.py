import os
import shutil

# create right folders
if not os.path.exists("./dataset_nosplit"):
    print("Making dataset_nosplit directory")
    os.makedirs("./dataset_nosplit")



if not os.path.exists(f"./dataset_nosplit/train"):
    print(f"Making train directory")
    os.makedirs(f"./dataset_nosplit/train")

if not os.path.exists(f"dataset_nosplit/test"):
    print(f"Making test directory")
    os.makedirs(f"dataset_nosplit/test")

if not os.path.exists(f"dataset_nosplit/train/HDR"):
    print(f"Making train HDR directory")
    os.makedirs(f"dataset_nosplit/train/HDR")
    
    if not os.path.exists(f"dataset_nosplit/train/LDR"):
        print(f"Making train LDR directory")
        os.makedirs(f"dataset_nosplit/train/LDR")
    
if not os.path.exists(f"dataset_nosplit/test/LDR"):
    print(f"Making test LDR directory")
    os.makedirs(f"dataset_nosplit/test/LDR")
    
if not os.path.exists(f"dataset_nosplit/test/HDR"):
    print(f"Making test HDR directory")
    os.makedirs(f"dataset_nosplit/test/HDR")


# move the data to the right directory
data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/video_match128/video_match128"

list_ldr = [fn for fn in os.listdir(data_path) if fn.endswith('_video.png')]
print("LDR images:", len(list_ldr))
list_hdr = [fn for fn in os.listdir(data_path) if fn.endswith('_ref.hdr')]
print("Clear:", len(list_hdr))

ldr_train = list_ldr[3:]
ldr_test = list_ldr[:3]
hdr_train = list_hdr[3:]
hdr_test = list_hdr[:3]


for filename in ldr_train:
    src_path = os.path.join(data_path, filename)
    dst_path = f"./dataset_nosplit/train/LDR"
    if not(os.path.isfile(dst_path)):
        shutil.copy(src_path, dst_path)

for filename in ldr_test:
    src_path = os.path.join(data_path, filename)
    dst_path = f"./dataset_nosplit/test/LDR"
    if not(os.path.isfile(dst_path)):
        shutil.copy(src_path, dst_path)

for filename in hdr_train:
    src_path = os.path.join(data_path, filename)
    dst_path = f"./dataset_nosplit/train/HDR"
    if not(os.path.isfile(dst_path)):
        shutil.copy(src_path, dst_path)

for filename in hdr_test:
    src_path = os.path.join(data_path, filename)
    dst_path = f"./dataset_nosplit/test/HDR"
    if not(os.path.isfile(dst_path)):
        shutil.copy(src_path, dst_path)


    print(f"{len(ldr_train)} training images and {len(ldr_test)} testing images")