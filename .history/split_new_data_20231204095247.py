import os
import shutil

data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/all_datasets"
sub_folders = ['train', 'test']
sub_sub_folders = ['HDR', 'LDR']

# create right folders
if not os.path.exists("./dataset_final"):
    print("Making final dataset directory")
    os.makedirs("./dataset_final")

for sub_folder in sub_folders:
if not os.path.exists(f"./dataset_final/test"):
    print(f"Making testing directory")
    os.makedirs(f"./dataset_final/test")


if not os.path.exists(f"./dataset_final/train"):
    print(f"Making training directory")
    os.makedirs(f"./dataset_final/train")

if not os.path.exists(f"./dataset_final/test"):
    print(f"Making testing directory")
    os.makedirs(f"./dataset_final/test")