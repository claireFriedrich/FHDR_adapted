import numpy as np
import os

data_path = "C:\Users\fricl\OneDrive\Documents\Suisse\EPFL\Cours\MA1\ML\video_match128"

if not os.path.exists("./dataset/mixed"):
            print("Making mixed directory")
            os.makedirs("./mixed")

if not os.path.exists("./dataset/clear"):
            print("Making clear directory")
            os.makedirs("./clear")


if not os.path.exists("./overcast"):
            print("Making overcast directory")
            os.makedirs("./overcast")