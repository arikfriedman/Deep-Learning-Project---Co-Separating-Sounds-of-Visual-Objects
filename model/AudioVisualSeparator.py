from torch import nn
import torch
from .Visual import Visual
from .UNet7Layer import UNet7Layer
from .Classifier import Classifier
from torch.autograd import Variable
import torchvision.transforms as T
import numpy as np

class AudioVisualSeparator(nn.Module):
    def __init__(self):
        super(AudioVisualSeparator, self).__init__()
        self.visual = Visual().create_visual_vector()
        self.uNet7Layer = UNet7Layer()
        self.classifier = Classifier()  #for weak labels

    '''X is the input and will in a format of a dictionary with several entries'''
    def forward(self, X):
        videos = [X['obj1']['id']] + [X['obj2']['id']]
        self_audios = [X['obj1']['audio']['stft'][0], X['obj2']['audio']['stft'][0]]  #array includes both videos data - 2 values
        self_audios = np.vstack(self_audios)

        detected_objects = [T.ToTensor()(c[1]).unsqueeze(0) for c in X['obj1']['images'][:]] +\
                           [T.ToTensor()(c[1]).unsqueeze(0) for c in X['obj2']['images'][:]]    #all detected objects in both video's'
        num_objs = len(detected_objects)
        mixed_audio = []
        mix = X['mix'][0]
        for n in range(num_objs):
            mixed_audio.append(torch.FloatTensor(mix).unsqueeze(0))
        mixed_audio = np.vstack(mixed_audio)
        mixed_audio = mixed_audio + 1e-10  # in order to make sure we don't divide by 0
        mixed_audio = torch.from_numpy(mixed_audio)

        detected_objects = np.vstack(detected_objects)
        detected_objects = torch.from_numpy(detected_objects)
        #detected_objects = T.ToTensor()(detected_objects)
        weak_labels = [c[0] for c in X['obj1']['images'][:]] + [c[0] for c in X['obj2']['images'][:]]              #a label per detected object
        log_mixed_audio = torch.log(mixed_audio).detach()

        ''' mixed audio and audio are after STFT '''
        
        # mask for the object
        ground_mask = self_audios / mixed_audio     #list of masks per video 
        #should we clamp ? - mask = mask.clamp(0, 5)

        # Resnet18 for the visual part of the detected object
        visual_vecs = self.visual(Variable(detected_objects, requires_grad=False))

        # should we use here the forward func?
        mask_preds = self.uNet7Layer(log_mixed_audio, visual_vecs)

        masks_applied = mask_preds * mixed_audio

        # after this there will be an iSTFT in the next layer of the net if we would like to hear the sound...

        audio_label_preds = self.classifier(torch.log(masks_applied + 1e-10))

        return {"ground_masks" : ground_mask, "predicted_audio_labels" : audio_label_preds, "predicted_masks" : mask_preds, "predicted_spectrograms" : masks_applied,
                "visual_objects" : visual_vecs, "mixed_audios" : mixed_audio}#, "videos" : videos}

'''https://github.com/rhgao/co-separation/blob/bd4f4fd51f2d6090e1566d20d4e0d0c8d83dd842/models/audioVisual_model.py'''
