from torch import nn
from ResNet18 import ResNet18

class AudioVisualSeparator(nn.Module):
    def __init__(self):
        super(AudioVisualSeparator, self).__init__()

        self.RN18 = ResNet18()