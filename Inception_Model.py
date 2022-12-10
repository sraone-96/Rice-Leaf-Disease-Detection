import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn, optim
# from torchsummary import summary
import torch.nn.functional as F

class Inception_Model(nn.Module):
	def __init__(self):
		super(Inception_Model,self).__init__()

		self.model_pretrained_first = models.inception_v3(pretrained=True)
		self.model_pretrained_first = nn.Sequential(*list(self.model_pretrained_first.children())[0:15])

		self.model_pretrained_second = models.inception_v3(pretrained=True)
		self.model_pretrained_second = nn.Sequential(*list(self.model_pretrained_second.children())[16:-3])

		# print(summary(self.model_pretrained,(3, 299, 299) ))
		for param in self.model_pretrained_first.parameters():
			param.requires_grad=False

		for param in self.model_pretrained_second.parameters():
			param.requires_grad=False


		self.internal_conv = nn.Conv2d(32, 1, 142, stride=1)
		self.conv1 = nn.Sequential(*list(models.inception_v3(pretrained=True).children())[0:1])
		for param in self.conv1.parameters():
			param.requires_grad=False
		# print(summary(self.conv1,(3, 299, 299) ))
		
		self.fc = nn.Linear(2049, 4)
		self.attention = nn.TransformerEncoderLayer(d_model = 64, nhead=1, batch_first=True)
		self.pool = nn.AvgPool2d(8,1)
		
			
	def forward(self,x):
		# print(x.shape)
		out_intermediate = self.model_pretrained_first(x)
		out1 = self.model_pretrained_second(out_intermediate)
		# print(out1.shape)
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
		# print(final_input.shape)
		avgpool_ouput = self.pool(final_input)
		# print(avgpool_ouput.shape)
		final_input = torch.squeeze(torch.squeeze(avgpool_ouput,2),2)
		final_output = self.fc(final_input)
		
		# print('-'*50)
		# print(final_output.shape)
		return final_output