from torch import nn
from Visual import Visual
from UNet7Layer import UNet7Layer
from Classifier import Classifier

class AudioVisualSeparator(nn.Module):
    def __init__(self, nets):
        super(AudioVisualSeparator, self).__init__()
        self.visual = Visual().create_visual_vector()
        self.uNet7Layer = UNet7Layer()
        self.classifier = Classifier()  #for weak labels

    '''X is the input and will in a format of a dictionary with several entries'''
    def forward(self, X):
        self_audio = X["audio"]
        mixed_audio = X["mixed_audio"] + 1e-8    # in order to make sure we don't divide by 0
        detected_object = X["detected_object"]
        weak_labels = X["weak_lebels"]

        ''' mixed audio and audio are after STFT '''
        
        # mask for the object
        ground_mask = self_audio / mixed_audio
        #should we clamp ? - mask = maks.clamp(0, 5)

        # Resnet18 for the visual part of the detected object
        visual_vec = self.visual(detected_object)

        # should we use here the forward func?
        mask_pred = self.uNet7Layer(mixed_audio, visual_vec)

        mask_applied = mask_pred * mixed_audio

        # after this there will be an iSTFT in the next layer of the net if we would like to hear the sound...

        audio_label_pred = self.classifier(mask_applied)

        return {"ground_mask" : ground_mask, "audio_label" : audio_label_pred, "audio_separated" : mask_applied}
