import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn, optim
import torch.nn.functional as F

class ResNet_Model(nn.Module):
	def __init__(self):
		super(ResNet_Model,self).__init__()

		# return_nodes = {
		# "fc" : "layer_final"
		# }
		# return_nodes2 = {
		# "Conv2d_1a_3x3.conv" : "layer_initial_conv"
		# }

		self.model_pretrained = models.resnet101(pretrained=True)
		self.model_pretrained = nn.Sequential(*list(self.model_pretrained.children())[:-2])

		# print(summary(self.model_pretrained,(3, 299, 299) ))
		for param in self.model_pretrained.parameters():
			param.requires_grad=False
		
		self.internal_conv = nn.Conv2d(64, 1, 141, stride=1)
		# self.conv1 = create_feature_extractor(models.inception_v3(pretrained=True), return_nodes=return_nodes2)
		self.conv1 = nn.Sequential(*list(models.resnet101(pretrained=True).children())[0:1])
		for param in self.conv1.parameters():
			param.requires_grad=False
		# print(summary(self.conv1,(3, 299, 299) ))
		
		self.fc = nn.Linear(2049, 4)
		self.attention = nn.TransformerEncoderLayer(d_model = 100, nhead=1, batch_first=True)
		self.pool = nn.AvgPool2d(10,1)
		
			
	def forward(self,x):
		# print(x.shape)
		out1 = self.model_pretrained(x)
		conv1_output = (self.conv1(x))
		# print(conv1_output.shape)
		internal_conv_out = self.internal_conv(conv1_output)
		
		# print(out1.shape, internal_conv_out.shape)
		x = internal_conv_out.shape
		attention_input = torch.reshape(internal_conv_out, (internal_conv_out.shape[0],internal_conv_out.shape[1],-1))
		# print(attention_input.shape)
		attention_output = self.attention(attention_input)
		# print(attention_output.shape)
		final_input =  torch.cat((out1, attention_output.reshape(*x)),dim=1)
		avgpool_ouput = self.pool(final_input)
		# print(avgpool_ouput.shape)
		final_input = torch.squeeze(torch.squeeze(avgpool_ouput,2),2)
		final_output = self.fc(final_input)
		
		# print('-'*50)
		# print(final_output.shape)
		return final_output