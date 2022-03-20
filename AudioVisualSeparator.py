from torch import nn
from Visual import Visual
from UNet7Layer import UNet7Layer
from Classifier import Classifier

class AudioVisualSeparator(nn.Module):
    def __init__(self, nets):
        super(AudioVisualSeparator, self).__init__()
        self.visual = Visual()
        self.uNet7Layer = UNet7Layer()
        self.classifier = Classifier()

    def forward(self, x):
        