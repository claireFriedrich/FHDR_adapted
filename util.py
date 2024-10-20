"""Some utility functions"""

import os

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt


def load_checkpoint(model, ckpt_path):
    """
    Load checkpoints for continuing training or evaluation.
    Loads the saved model state from the checkpoint path.
    """
    start_epoch = np.loadtxt("./checkpoints/state.txt", dtype=int)
    model.load_state_dict(torch.load(ckpt_path))
    print("Resuming from epoch ", start_epoch)
    return start_epoch, model

def make_required_directories(mode):
    """
    Create necessary directories for training or testing.
    Create specific directories based on the mode provided: 'train' or 'test'.
    """
    print(f"Mode = {mode}")
    if mode == "train":
        if not os.path.exists("./checkpoints"):
            print("Making checkpoints directory")
            os.makedirs("./checkpoints")

        if not os.path.exists("./training_results"):
            print("Making training_results directory")
            os.makedirs("./training_results")
    elif mode == "test":
        if not os.path.exists("./test_results"):
            print("Making test_results directory")
            os.makedirs("./test_results")

# computes the tone mapping images of the GT HDR image.
def mu_tonemap(img):
    """
    Tonemap HDR images using μ-law before computing loss.
    Apply μ-law transformation to the input image tensor.
    """
    MU = 5000.0
    return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)


def write_hdr(hdr_image, path):
    """
    Write the HDR image in radiance format (.hdr) to the specified path.
    """
    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    print(f"norme :{norm_image.max() - norm_image.min()}")
    with open(path, "wb") as f:
        norm_image = (norm_image - norm_image.min()) / (
            norm_image.max() - norm_image.min()
        )  # normalisation function
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (norm_image.shape[0], norm_image.shape[1]))
        brightest = np.maximum(
            np.maximum(norm_image[..., 0], norm_image[..., 1]), norm_image[..., 2]
        )
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(norm_image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)
        f.close()


def save_hdr_image(img_tensor, batch, path):
    """
    Pre-process the HDR image tensor before writing.
    Save the HDR image tensor after necessary pre-processing steps.
    """
    img = img_tensor.data[batch].cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))

    write_hdr(img.astype(np.float32), path)


def save_ldr_image(img_tensor, batch, path):
    """
    Pre-process and saves the LDR image tensor to the specified path.
    """
    img = img_tensor.data[batch].cpu().float().numpy()
    img = 255 * (np.transpose(img, (1, 2, 0)) + 1) / 2

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def save_checkpoint(epoch, model):
    """
    Save the model's current state and epoch number as a checkpoint.
    """
    checkpoint_path = os.path.join("./checkpoints", "epoch_" + str(epoch) + ".ckpt")
    latest_path = os.path.join("./checkpoints", "latest.ckpt")
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    np.savetxt("./checkpoints/state.txt", [epoch + 1], fmt="%d")
    print("Saved checkpoint for epoch ", epoch)


def update_lr(optimizer, epoch, opt):
    """
    Adjust the learning rate of the optimizer after a certain number of epochs.
    """
    new_lr = opt.lr - opt.lr * (epoch - opt.lr_decay_after) / (
        opt.epochs - opt.lr_decay_after
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    print("Learning rate decayed. Updated LR is: %.6f" % new_lr)


def plot_losses(training_losses, validation_losses, num_epochs, path):
    """
    Plot the training losses in function of the number of epochs.
    """
    plt.figure()
    plt.plot(np.linspace(1, num_epochs, num=num_epochs), training_losses, label="training")
    plt.plot(np.linspace(1, num_epochs, num=num_epochs), validation_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(os.path.basename(os.path.normpath(path)))
    plt.legend()
    plt.savefig(path)