import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from model.AudioVisualSeparator import AudioVisualSeparator

if __name__ == "__main__":
    mod = AudioVisualSeparator()
    for param in mod.parameters():
        print(param)
    mod.forward(X)





    #path = r"C:\Users\user\Desktop\etc\chunk_10\cropped_000011\39.jpg"

    '''
    im = Image.open(path).resize((224, 224))
    #im.show()

    tim = T.ToTensor()(im)
    #tim.res(3*224*224)
#    print(torch.max(tim[0]))
    tsfm = T.Normalize(mean=(0.1057, 0.1525, 0.1557), std=(0.0953, 0.1483, 0.1151))
    lis = tsfm(tim)
    nor = T.ToPILImage()(lis)
    #nor.show()


    pix = np.asarray(im).astype('float32')
    a = [pix]
    a += [pix]
    a = [np.ones((2, 3))]
    a[0][0][0] = 9
#    a = T.ToTensor()(np.asarray([tim, tim]))
    a = (tim + tim) / 2
    a -= tim
    print(a)
    print("ANS:")
    print(a.std((1, 2)))
    #print(a.mean())
    #print(np.std(a[0][1][:][:]))
    #print(np.std(a[0][:][:][2]))
    #print(a[0][3])
    #print(len(a[0]))
    '''