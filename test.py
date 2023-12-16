import numpy as np
import torch
import torch.nn as nn
from skimage.measure import compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import HDRDataset
from model import FHDR
from options import Options
from util import make_required_directories, mu_tonemap, save_hdr_image, save_ldr_image

# initialise options
opt = Options().parse()
opt.log_scores = True

# print the configured options
print(opt)

# ======================================
# Load the data
# ======================================

dataset = HDRDataset(mode="test", opt=opt)
data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# print the number of testing images
print("Testing samples: ", len(dataset))

# ========================================
# Model initialisation, 
# loading & GPU configuration
# ========================================

# get GPU IDs from optiont
str_ids = opt.gpu_ids.split(",")
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set GPU device
if torch.cuda.is_available():
    print(f"#GPUs = {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        print(f"GPU {i} Name:", torch.cuda.get_device_name(device))
else:
    print("No GPU available.")
    device = torch.device("cpu")
    print(f"CPU: {device}")


# initialize FHDR model with specified iterations
model = FHDR(iteration_count=opt.iter, device=device)
model.to(device)

# define the mean squared error loss
mse_loss = nn.MSELoss()

# load the checkpoint for the evaluation
model.load_state_dict(torch.load(opt.ckpt_path, map_location=device))

# make the necessary directories for saving the test results
make_required_directories(mode="test")

# initialize the evaluation metrics
avg_psnr = 0
avg_ssim = 0
avg_mse = 0

print("Starting evaluation. Results will be saved in '/test_results' directory")

# ========================================
# Evaluation of the model, 
# computation of the evaluation metrics
# ========================================

with torch.no_grad():

    for batch, data in enumerate(tqdm(data_loader, desc="Testing %")):

        # get the LDR images
        input = data["ldr_image"].data.to(device)
        # get the HDR images
        ground_truth = data["hdr_image"].data.to(device)

        # generate the output from the model
        output = model(input)

        # tonemap the ground truth image for PSNR-μ calculation
        mu_tonemap_gt = mu_tonemap(ground_truth)

        # get the final output from the model
        output = output[-1]

        for batch_ind in range(len(output.data)):

            # save the generated images
            save_ldr_image(img_tensor=input, batch=batch_ind, path="./test_results/ldr_b_{}_{}.png".format(batch, batch_ind),)
            
            save_hdr_image(img_tensor=output, batch=batch_ind, path="./test_results/generated_hdr_b_{}_{}.hdr".format(batch, batch_ind),)
            
            save_hdr_image(img_tensor=ground_truth, batch=batch_ind, path="./test_results/gt_hdr_b_{}_{}.hdr".format(batch, batch_ind),)

            if opt.log_scores:
                # calculate the PSNR score
                mse = mse_loss(mu_tonemap(output.data[batch_ind]), mu_tonemap_gt.data[batch_ind])
                avg_mse += mse.item()
                psnr = 10 * np.log10(1 / mse.item())

                avg_psnr += psnr

                generated = (np.transpose(output.data[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                real = (np.transpose(ground_truth.data[batch_ind].cpu().numpy(), (1, 2, 0))+ 1) / 2.0

                # calculate the SSIM score
                ssim = compare_ssim(generated, real, multichannel=True)
                avg_ssim += ssim

# ========================================
# Printing the results
# ========================================

if opt.log_scores:
    print("===> Avg PSNR: {:.4f} dB".format(avg_psnr / len(dataset)))
    print("Avg SSIM -> " + str(avg_ssim / len(dataset)))
    print("Avg MSE -> " + str(avg_mse / len(dataset)))

print("Evaluation completed.")
