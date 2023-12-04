import numpy as np
import os

data_path = "Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/video_match128"

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