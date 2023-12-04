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
        os.makedirs("./dataset/mixed")

if not os.path.exists("./dataset/clear"):
    print("Making clear directory")
    os.makedirs("./dataset/clear")

if not os.path.exists("./dataset/overcast"):
    print("Making overcast directory")
    os.makedirs("./dataset/overcast")


for folder in folders: 
    if not os.path.exists("./dataset/overcast/train"):
        print("Making overcast/train directory")
        os.makedirs("./dataset/overcast")