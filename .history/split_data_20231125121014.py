import numpy as np
import os

data_path = "Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/video_match128"

if not os.path.exists("./dataset"):
            print("Making dataset directory")
            os.makedirs("./dataset")

if not os.path.exists("./dataset/mixed"):
            print("Making mixed directory")
            os.makedirs("./dataset/mixed")

if not os.path.exists("./dataset/clear"):
            print("Making clear directory")
            os.makedirs("./dataset/clear")

if not os.path.exists("./dataset/overcast"):
            print("Making overcast directory")
            os.makedirs("./dataset/overcast")

folders = ['']
if not os.path.exists("./dataset/overcast/train"):
            print("Making overcast/train directory")
            os.makedirs("./dataset/overcast")