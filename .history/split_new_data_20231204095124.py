import os
import shutil

data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/all_datasets"

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

subfolder = ['LDR', 'HDR']
if not os.path.exists(f"./dataset_final/test"):
    print(f"Making testing directory")
    os.makedirs(f"./dataset_final/test")