import torch
import torch.nn as nn

from encoders.sketchrnn import SketchRNNEmbedding
from encoders.clip_encoder import CLIP_VITB16
from encoders.shared_mlp import SharedMLP

from options import Option
from .rn import Relation_Network, cos_similar

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.args = args
        self.sketch_rnn = SketchRNNEmbedding(enc_hidden_size=256)
        self.clip = CLIP_VITB16()
        self.shared_mlp = SharedMLP(512, 512, 512)
        self.rn = Relation_Network(args.anchor_number, dropout=0.1)
        
    def forward(self, sk, im, stage='train', only_self=False):
        if stage == "train":
            sk = self.sketch_rnn(sk)
            sk = self.shared_mlp(sk)
            
            im = self.clip.encode_image(im)
            im = self.shared_mlp(im)
            
            fea = torch.cat([sk, im], dim=0)  
                
            return fea
        else:
            if only_self:
                if sk is None:
                    im = self.clip.encode_image(im)
                    self_fea = self.shared_mlp(im)
                    return self_fea
                else:
                    sk = self.sketch_rnn(sk)
                    self_fea = self.shared_mlp(sk)
                    return self_fea
                
            else:
                sk = self.sketch_rnn(sk)
                sk = self.shared_mlp(sk)
            
                im = self.clip.encode_image(im)
                im = self.shared_mlp(im)
            
                fea = torch.cat([sk, im], dim=0) 
                
                return fea


if __name__ == '__main__':
    args = Option().parse()
    sk = torch.rand((4, 1024, 5))
    im = torch.rand((4, 3, 224, 224))
    model = Model(args)
    cls_fea = model(sk, im)
    print(f"sketch features: {cls_fea.shape}")
