"""
Script for training the FHDR model.
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from data_loader import HDRDataset
from model import FHDR
from options import Options
# VGGNet with 19 convolutional layers
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,
    plot_losses
)
# where they define the model 
# VGG = classical/standard convolutional neural network architecture. 3x3 filters. SImple model. Just pooling, convolutional layers and 1 fully connected layer.
# Visual Geometry Group = university of oxford, the company that created the VGGNet, for image classification. 
# here they use VGG19 --> 19 convolutional layers
from vgg import VGGLoss

from sklearn.model_selection import train_test_split


if not os.path.exists(f"./plots"):
        print(f"Making plot directory")
        os.makedirs(f"./plots")
def weights_init(m):
    """
    Initializing the weights of the network as a first step.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # std 0.0 -> runtime error so changed to 0.01
        m.weight.data.normal_(0.0, 0.01)


# initialise training options
opt = Options().parse()
opt.save_results_after = 400
opt.log_after = 1

# ======================================
# loading data
# ======================================

dataset = HDRDataset(mode="train", opt=opt)

# split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# create separate data loaders for training and validation
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

print("Training samples in one batch: ", len(train_data_loader) * opt.batch_size)
print("Validation samples in one batch: ", len(val_data_loader) * opt.batch_size)


# ========================================
# gpu configuration
# ========================================

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

# ========================================
# model init
# ========================================

model = FHDR(iteration_count=opt.iter, device=device)
model.to(device)

# ========================================
#  initialising losses and optimizer
# ========================================

l1 = torch.nn.L1Loss()
# VGGloss of the VGGNet
perceptual_loss = VGGLoss(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

# create directories to save the training results
make_required_directories(mode="train")

# ==================================================
#  loading checkpoints if continuing training
# ==================================================
print(opt)

if opt.continue_train:
    try:
        start_epoch, model = load_checkpoint(model, opt.ckpt_path)

    except Exception as e:
        print(e)
        print("Checkpoint not found! Training from scratch.")
        start_epoch = 1
        model.apply(weights_init)
else:
    start_epoch = 1
    model.apply(weights_init)

if opt.print_model:
    print(model)

# ========================================
#  training
# ========================================
    
# define the number of epochs
num_epochs = 200

print(f"# of epochs: {num_epochs}")

losses_train = []
losses_validation = []

# epoch = one complete pass of the training dataset through the algorithm
for epoch in range(start_epoch, num_epochs + 1):
    print(f"-------------- Epoch # {epoch} --------------")

    epoch_start = time.time()

    # check whether learning rate needs to be updated
    if epoch > opt.lr_decay_after:
        update_lr(optimizer, epoch, opt)

    losses_epoch = []
    vgg_losses_epoch = []

    # training loop 
    # stochstic gradient descent with batch size = 2
    for batch, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        # get the LDR images
        input = data["ldr_image"].data.to(device)
        # get the HDR images
        ground_truth = data["hdr_image"].data.to(device)

        # forward pass 
        output = model(input)

        l1_loss = 0
        vgg_loss = 0

        # tonemapping ground truth
        mu_tonemap_gt = mu_tonemap(ground_truth)

        # computing loss for n generated outputs (from n-iterations)
        for image in output:
            l1_loss += l1(mu_tonemap(image), mu_tonemap_gt)
            vgg_loss += perceptual_loss(mu_tonemap(image), mu_tonemap_gt)

        # averaged over n iterations (n output images for 1 input)
        l1_loss /= len(output)
        vgg_loss /= len(output)

        # averaged over batches
        l1_loss = torch.mean(l1_loss)
        vgg_loss = torch.mean(vgg_loss)


        # FHDR loss function
        loss = l1_loss + (vgg_loss * 10)
        losses_epoch.append(loss.item())
        

        # output is the final reconstructed image i.e. last in the array of outputs of n iterations
        output = output[-1]

        # backpropagate and step
        loss.backward()
        optimizer.step()


        # save image results
        if (batch + 1) % opt.save_results_after == 0:
            print(f"Batch = {batch}")
            print(data["path"])
            print("training_results/generated_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1))

            save_ldr_image(
                img_tensor=input,
                batch=0,
                path="./training_results/ldr_e_{}_b_{}.jpg".format(epoch, batch + 1),
            )
            save_hdr_image(
                img_tensor=output,
                batch=0,
                path="./training_results/generated_hdr_e_{}_b_{}.hdr".format(
                    epoch, batch + 1
                ),
            )
            save_hdr_image(
                img_tensor=ground_truth,
                batch=0,
                path="./training_results/gt_hdr_e_{}_b_{}.hdr".format(epoch, batch + 1),
            )

    print(f"Training loss: {losses_epoch[-1]}")
    losses_train.append(losses_epoch[-1])

# ========================================
# Validation
# ========================================
    
    # set model to evaluation mode
    model.eval()  
    val_losses = []
    with torch.no_grad():
        for val_batch, val_data in enumerate(val_data_loader):
            input_val = val_data["ldr_image"].data.to(device)
            ground_truth_val = val_data["hdr_image"].data.to(device)

            output_val = model(input_val)

            # calculate validation loss 
            l1_loss_val = 0
            vgg_loss_val = 0
            mu_tonemap_gt_val = mu_tonemap(ground_truth_val)

            for image_val in output_val:
                l1_loss_val += l1(mu_tonemap(image_val), mu_tonemap_gt_val)
                vgg_loss_val += perceptual_loss(mu_tonemap(image_val), mu_tonemap_gt_val)

            l1_loss_val /= len(output_val)
            vgg_loss_val /= len(output_val)

            l1_loss_val = torch.mean(l1_loss_val)
            vgg_loss_val = torch.mean(vgg_loss_val)

            val_loss = l1_loss_val + (vgg_loss_val * 10)
            val_losses.append(val_loss.item())

    # Calculate average validation loss for the entire validation dataset
    average_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average validation Loss: {average_val_loss}")
    losses_validation.append(average_val_loss)

    # set model back to training mode
    model.train()  


    epoch_finish = time.time()
    time_taken = (epoch_finish - epoch_start)

    print("End of epoch {}. Time taken: {} s.".format(epoch, int(time_taken)))

    if epoch % opt.save_ckpt_after == 0:
        save_checkpoint(epoch, model)

print("Training complete!")

print(f"Training losses: {losses_train}")
print(f"Validation losses: {losses_validation}")

plot_losses(losses_train, losses_validation, num_epochs, f"plots/_loss_vgg_{num_epochs}_epochs")
