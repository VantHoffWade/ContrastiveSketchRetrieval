import torch
from torch import nn

class SharedMLP(nn.Module):
	def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.1,
				 activation=nn.ReLU(inplace=True)):
		super().__init__()
		self.fc1 = nn.Linear(in_channels, hidden_channels)
		self.fc2 = nn.Linear(hidden_channels, out_channels)
		self.dropout = nn.Dropout(dropout)
		self.activation = activation

	def forward(self, x):
		x = self.activation(self.fc1(x))
		x = self.fc2(self.dropout(x))
		return x
