import numpy as np
import os

data_path = "C:\Users\fricl\OneDrive\Documents\Suisse\EPFL\Cours\MA1\ML\video_match128"

if not os.path.exists("./"):
            print("Making checkpoints directory")
            os.makedirs("./checkpoints")