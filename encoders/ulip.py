import timm
import torch.nn as nn
import torch
from collections import OrderedDict

class ImageEncoder_ULIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
        self.image_projection = nn.Parameter(torch.empty(768, 512))

    @torch.inference_mode()
    def forward(self, image):
        x = self.vision_model(image)
        x = x @ self.image_projection

        return x

def test():
    amodel = ImageEncoder_ULIP()
    emb = torch.rand(9, 3, 224, 224)

    out = amodel(emb)
    print(out)


if __name__ == '__main__':
    # test()

    # save_weights_from_all()
    test()

    pass


