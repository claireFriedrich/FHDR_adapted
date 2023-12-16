import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FHDR(nn.Module):
    """
    Class for a Fast High Dynamic Range (FHDR) model.
    """
    def __init__(self, iteration_count, device):
        """
        Initializes the FHDR model.
        """
        # gives access to methods in a superclass from the subclass that inherits from it
        super(FHDR, self).__init__() 
        print("FHDR model initialised")

        self.device = device

        self.iteration_count = iteration_count

        # layers for initial feature extraction
        self.reflect_pad = nn.ReflectionPad2d(1)
        self.feb1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.feb2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # feedback block for iterative processing
        self.feedback_block = FeedbackBlock(device=self.device)

        # layers for high-resolution reconstruction
        self.hrb1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.hrb2 = nn.Conv2d(64, 3, kernel_size=3, padding=0)

        # final output transformation 
        self.tanh = nn.Tanh()

    def forward(self, input):
        """
        Defines the forward pass of the model. 
        """
        outs = []

        feb1 = F.relu(self.feb1(self.reflect_pad(input)))
        feb2 = F.relu(self.feb2(feb1))

        for i in range(self.iteration_count):
            fb_out = self.feedback_block(feb2)

            # combining feedback and initial features
            FDF = fb_out + feb1

            hrb1 = F.relu(self.hrb1(FDF))
            out = self.hrb2(self.reflect_pad(hrb1))
            out = self.tanh(out)
            outs.append(out)

        return outs


class FeedbackBlock(nn.Module):
    """
    Class for a feedback block that maintains the state across iterations.
    """
    def __init__(self, device):
        """
        Initializes the feedback block that retains state across iterations.
        """
        super(FeedbackBlock, self).__init__()

        self.device = device

        self.compress_in = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.DRDB1 = DilatedResidualDenseBlock(device=device)
        self.DRDB2 = DilatedResidualDenseBlock(device=device)
        self.DRDB3 = DilatedResidualDenseBlock(device=device)
        self.last_hidden = None

        self.GFF_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.should_reset = True

    def forward(self, x):
        """
        Forward pass of the feedback block.
        """
        if self.should_reset:
            # initialize the hidden state for the feedback
            self.last_hidden = torch.zeros(x.size()).to(self.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        out1 = torch.cat((x, self.last_hidden), dim=1)
        out2 = self.compress_in(out1)

        out3 = self.DRDB1(out2)
        out4 = self.DRDB2(out3)
        out5 = self.DRDB3(out4)

        out = F.relu(self.GFF_3x3(out5))
        self.last_hidden = out
        self.last_hidden = Variable(self.last_hidden.data)

        return out


class DilatedResidualDenseBlock(nn.Module):
    """
    Class for a dilated residual dense block.
    """
    def __init__(self, device, nDenselayer=4, growthRate=32):
        """
        Initializes the dilated residual dense block.
        """
        super(DilatedResidualDenseBlock, self).__init__()
        self.device = device

        nChannels_ = 64
        modules = []

        # creating multiple dense layers in the block
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.should_reset = True

        self.compress = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv_1x1 = nn.Conv2d(nChannels_, 64, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        """
        Forward pass of the dilated residual dense block.
        """
        if self.should_reset:
            # initialize the hidden state for the block
            self.last_hidden = torch.zeros(x.size()).to(self.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        cat = torch.cat((x, self.last_hidden), dim=1)

        out = self.compress(cat)
        out = self.dense_layers(out)
        out = self.conv_1x1(out)

        self.last_hidden = out
        self.last_hidden = Variable(out.data)

        return out


class make_dense(nn.Module):
    """
    CLass for a dense Connection in the Residual Block.
    """
    def __init__(self, nChannels, growthRate, kernel_size=3):
        """
        Initialize a dense connection in the residual block.
        """
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            bias=False,
            dilation=2,
        )

    def forward(self, x):
        """
        Forward pass of the dense connection.
        """
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
