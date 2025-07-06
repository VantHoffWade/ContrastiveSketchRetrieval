import torch
import torch.nn as nn

from encoders.sketchrnn import SketchRNNEmbedding
from encoders.clip_encoder import CLIP_VITB16
from encoders.shared_mlp import SharedMLP

from options import Option

class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()

		self.args = args

		self.sketch_rnn = SketchRNNEmbedding(enc_hidden_size=256)
		self.clip = CLIP_VITB16()
		self.shared_mlp = SharedMLP(512, 512, 512)
	def forward(self, sk, im):
		sketch_fea = self.sketch_rnn(sk)
		image_fea = self.clip.encode_image(im)

		refactor_sketch_fea = self.shared_mlp(sketch_fea)
		refactor_image_fea = self.shared_mlp(image_fea)
		fea = torch.cat([refactor_sketch_fea, refactor_image_fea], dim=0)

		return fea


if __name__ == '__main__':
	args = Option().parse()
	sk = torch.rand((4, 1024, 5))
	im = torch.rand((4, 3, 224, 224))
	model = Model(args)
	cls_fea = model(sk, im)
	print(f"sketch features: {cls_fea.shape}")
