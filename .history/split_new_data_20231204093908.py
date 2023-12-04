import os
import shutil

# create right folders
if not os.path.exists("./dataset_final"):
    print("Making final dataset directory")
    os.makedirs("./dataset_final")


if not os.path.exists(f"./dataset/test"):
    print(f"Making {folder} directory")
    os.makedirs(f"./dataset/{folder}")