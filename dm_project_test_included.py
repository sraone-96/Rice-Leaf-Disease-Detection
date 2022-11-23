# -*- coding: utf-8 -*-
"""DM-Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zV4Fvl8SCGWtP1gIXmodHDXtBPunGLOS
"""

import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn, optim
import torch.nn.functional as F

class myModel(nn.Module):
	def __init__(self):
		super(myModel,self).__init__()

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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from collections import Counter
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
from torchsummary import summary
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.metrics import classification_report
import os
from pathlib import Path



# seed = 0
use_cuda=0
# torch.manual_seed(seed)
transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.ToTensor()
])

input_shape = [299,299,3]

batch_size = 32
dl_args = dict(batch_size=batch_size)
data_dir = './Rice Leaf Disease Images'

# Split the dataset into train, validation and test
train_split = 0.7
valid_split = 0.1
test_split = 0.2
current_model = 'inception'
data = datasets.ImageFolder(data_dir, transform=transform)

n_val = int(np.floor(valid_split * len(data)))
n_test = int(np.floor(test_split * len(data)))
n_train = len(data) - n_val - n_test

# # train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
# train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test], generator=torch.Generator())

# print(dict(Counter(data.targets)))
# print(Counter(data.targets[i] for i in train_ds.indices)) 
# print(Counter(data.targets[i] for i in val_ds.indices)) 
# print(Counter(data.targets[i] for i in test_ds.indices)) 


# train_dl = DataLoader(train_ds, **dl_args)
# valid_dl = DataLoader(val_ds, **dl_args)
# test_dl = DataLoader(test_ds, **dl_args)

nclass = 4

# test_model = models.resnet101(pretrained=True)
# print(summary(test_model, (3, 299, 299)))
# test_model = nn.Sequential(*list(test_model.children())[:-2])

# # print(test_model)
# # print(test_model.Mixed_7c._get_name())
# # for child in test_model.children():
# # 			print(child.named_children)  # 3->6		


# print(summary(test_model, (3, 299, 299)))
# # print(get_graph_node_names(test_model))
model_transfer = myModel()
# exit()

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(filter(lambda p: p.requires_grad, model_transfer.parameters()), lr=0.01, momentum = 0.9)
softmax = nn.Softmax(dim=1)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, run_num):
    
    save_dir = './models_trained/'+  current_model + "/" + str(run_num)+'/'
    os.makedirs(save_dir, exist_ok=True)

    '''returns trained model'''
    # Initialize tracker for minimum validation loss
    valid_loss_min = np.inf
  
    for epoch in range(1, n_epochs+1):
        # Initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_predictions = []
        train_labels = []
        
        val_predictions = []
        val_labels = []
        print('-'*50)
        print('-'*50)
        # Model training
        model.train()
        for batch_idx, (data,target) in enumerate(loaders['train']):
            # Move to GPU
            if use_cuda:
                data,target= data.cuda(), target.cuda()
                model.cuda()
      
            # Clear the gradient of all optimized variables
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # print(output.shape)
            # Calculate the batch loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # removing tensors from gpu for space allocation
            output = torch.tensor(output, device = 'cpu')
            data = torch.tensor(data, device='cpu')
            target = torch.tensor(target, device='cpu')


            train_predictions.extend(torch.argmax(softmax(output),dim=1))
            train_labels.extend(target)
            
            torch.cuda.empty_cache()

            # Update the average validation loss
            train_loss = train_loss + ((1/ (batch_idx + 1 ))*(loss.data-train_loss))
        train_labels = torch.tensor(train_labels, device = 'cpu')
        train_predictions = torch.tensor(train_predictions, device = 'cpu')
        print(classification_report(train_labels,train_predictions))
        print('-'*50)
        
        # Model validation
        model.eval()
        for batch_idx, (data,target) in enumerate(loaders['valid']):
            # Move to GPU
            if use_cuda:
                # data, target = data.cuda(), target.cuda()
                data,target = data.cuda(), target.cuda()
                model.cuda()
            
            # Update the average validation loss
            # Forward pass: compute predicted outputs by passing inputs to the model
            
            output = model(data)
            # Calculate the batch loss
            loss = criterion(output, target)


            # removing tensors from gpu for space allocation
            output = torch.tensor(output, device = 'cpu')
            data = torch.tensor(data, device='cpu')
            target = torch.tensor(target, device='cpu')

            
            val_predictions.extend(torch.argmax(softmax(output),dim=1))
            val_labels.extend(target)
            
            torch.cuda.empty_cache()
            # Update the average validation loss
            valid_loss = valid_loss + ((1/ (batch_idx +1)) * (loss.data - valid_loss))

        # val_labels = torch.tensor(val_labels, device = 'cpu')
        # val_predictions = torch.tensor(val_predictions, device = 'cpu')
        print(classification_report(val_labels,val_predictions))
        print('-'*50)
        
        # print training/validation stats
        print('Epoch: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
            epoch,
            train_loss,
            valid_loss))
        # Save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(
                  valid_loss_min,
                  valid_loss))
            model_name = 'model_' + str(valid_loss.item())+'_'+str(epoch)+'.pt' 
            file_name = 'model_' + str(valid_loss.item())+'_'+str(epoch)+'.csv' 
            torch.save(model.state_dict(), save_dir + model_name)
            
            report = classification_report(val_labels, val_predictions, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(save_dir + file_name, sep='\t')
            valid_loss_min = valid_loss
  

    # Return trained model
    return model





def test(n_epochs, loaders, model, optimizer, criterion, use_cuda, run_num):
    
    
        # Model testing
        model.eval()
        test_labels = []
        test_predictions = []
        test_loss = 0.0
        for batch_idx, (data,target) in enumerate(loaders['test']):
            # Move to GPU
            if use_cuda:
                # data, target = data.cuda(), target.cuda()
                data,target = data.cuda(), target.cuda()
                model.cuda()
            
            output = model(data)
            # Calculate the batch loss
            loss = criterion(output, target)

            # removing tensors from gpu for space allocation
            output = torch.tensor(output, device = 'cpu')
            data = torch.tensor(data, device='cpu')
            target = torch.tensor(target, device='cpu')

            
            test_predictions.extend(torch.argmax(softmax(output),dim=1))
            test_labels.extend(target)
            
            torch.cuda.empty_cache()
            # Update the average training loss
            test_loss = test_loss + ((1/ (batch_idx +1)) * (loss.data - test_loss))

        
        print(classification_report(test_labels,test_predictions))
        file_name = 'test_results_' + str(run_num)+'.csv' 
        report = classification_report(test_labels,test_predictions, output_dict=True)
        df = pd.DataFrame(report).transpose()

        save_dir = './models_trained/'+  current_model + "/" + str(run_num)+'/'
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_dir + file_name, sep='\t')

        # print test stats
        print('\t Test Loss: {:.5f}'.format(
            test_loss))


# Train the model
seeds = range(0,30)
run_num=0
for ks in range(30):
    # print(seeds[ks])
    train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seeds[ks]))

    # print(dir(train_ds))
    # print(train_ds.indices)

    train_dl = DataLoader(train_ds, **dl_args)
    valid_dl = DataLoader(val_ds, **dl_args)
    test_dl = DataLoader(test_ds, **dl_args)


    # Define loaders transfer
    loaders_transfer = {'train': train_dl,
                        'valid': valid_dl,
                        'test': test_dl}


    model_transfer = train(100, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, run_num)
    model_best = myModel()
    dirs = "./models_trained/"  + current_model + "/" + str(run_num)
    paths = sorted(Path(dirs).iterdir(), key=os.path.getmtime)

    dt = torch.load(paths[0])
    model_best.load_state_dict(dt)
    test(0, loaders_transfer, model_best, optimizer_transfer, criterion_transfer, use_cuda, run_num)
    run_num = run_num+1
