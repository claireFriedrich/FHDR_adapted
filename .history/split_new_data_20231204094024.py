import os
import shutil

# create right folders
if not os.path.exists("./dataset_final"):
    print("Making final dataset directory")
    os.makedirs("./dataset_final")


if not os.path.exists(f"./dataset_final/test"):
    print(f"Making testing directory")
    os.makedirs(f"./dataset_final/test")


if not os.path.exists(f"./dataset_final/train"):
    print(f"Making training directory")
    os.makedirs(f"./dataset_final/train")

subfolder = ['']
if not os.path.exists(f"./dataset_final/test"):
    print(f"Making testing directory")
    os.makedirs(f"./dataset_final/test")