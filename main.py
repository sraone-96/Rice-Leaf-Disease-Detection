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
from myModel import myModel
from sklearn.metrics import classification_report


seed = 0
use_cuda=0
torch.manual_seed(seed)
transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.ToTensor()
])

input_shape = [299,299,3]

batch_size = 64
dl_args = dict(batch_size=batch_size)
data_dir = './Rice Leaf Disease Images/'

# Split the dataset into train, validation and test
train_split = 0.5
valid_split = 0.1
test_split = 0.4

data = datasets.ImageFolder(data_dir, transform=transform)
n_val = int(np.floor(valid_split * len(data)))
n_test = int(np.floor(test_split * len(data)))
n_train = len(data) - n_val - n_test

train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

print(dict(Counter(data.targets)))
print(Counter(data.targets[i] for i in train_ds.indices)) 
print(Counter(data.targets[i] for i in val_ds.indices)) 
print(Counter(data.targets[i] for i in test_ds.indices)) 


train_dl = DataLoader(train_ds, **dl_args)
valid_dl = DataLoader(val_ds, **dl_args)
test_dl = DataLoader(test_ds, **dl_args)

nclass = 4

model_transfer = myModel()
# print(summary(model_transfer, (3, 299, 299)))

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(filter(lambda p: p.requires_grad, model_transfer.parameters()), lr=0.01, momentum = 0.9)
softmax = nn.Softmax(dim=1)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda):
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
        # Model training
        model.train()
        for batch_idx, (data,target) in enumerate(loaders['train']):
            # Move to GPU
            if use_cuda:
                data,target = data.cuda(), target.cuda()
      
            # Clear the gradient of all optimized variables
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # print(output.shape)
            # print(output.shape)
            # Calculate the batch loss
            loss = criterion(output, target)
            train_predictions.extend(torch.argmax(softmax(output),dim=1))
            train_labels.extend(target)
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform a single optimization step (parameter update)
            optimizer.step()
            # Record the average training loss
            train_loss = train_loss + ((1/ (batch_idx + 1 ))*(loss.data-train_loss))
            break
        print("training done")
        print(classification_report(train_labels,train_predictions))
        # Model validation
        model.eval()
        for batch_idx, (data,target) in enumerate(loaders['valid']):
            # Move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # Update the average validation loss
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            val_predictions.extend(torch.argmax(softmax(output),dim=1))
            val_labels.extend(target)
            
            # Calculate the batch loss
            loss = criterion(output, target)
            # Update the average validation loss
            valid_loss = valid_loss + ((1/ (batch_idx +1)) * (loss.data - valid_loss))
            break
        print(classification_report(val_labels,val_predictions))
        
        # print training/validation stats
        print('Epoch: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
            epoch,
            train_loss,
            valid_loss))
    
        # Save the model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(
        #           valid_loss_min,
        #           valid_loss))
        #     torch.save(model.state_dict(), 'model_transfer.pt')
        #     valid_loss_min = valid_loss
  
    # Return trained model
    return model

# Define loaders transfer
loaders_transfer = {'train': train_dl,
                    'valid': valid_dl,
                    'test': test_dl}

# Train the model
model_transfer = train(10, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda)































# base_model = applications.InceptionV3(weights='imagenet', 
#                                 include_top=False, 
#                                 input_shape=(input_shape[0], input_shape[1],input_shape[2]))
# base_model.trainable = False

# add_model = Sequential()
# add_model.add(base_model)
# add_model.add(GlobalAveragePooling2D())
# add_model.add(Dropout(0.5))
# add_model.add(Dense(nclass, 
#                     activation='softmax'))

# model = add_model
# model.compile(loss='categorical_crossentropy', 
#               optimizer=optimizers.SGD(lr=1e-4, 
#                                        momentum=0.9),
#               metrics=['accuracy'])
# model.summary()




# model_transfer = mymodel()
# model_transfer = models.inception_v3(pretrained=True)
# # print(get_graph_node_names(model_transfer))

# # for child in model_transfer.children():
# # 			print(child)  # 3->6		

# return_nodes = {
#     # "layer4.2.conv3": "layer4"
#     "Mixed_7c.branch_pool.conv" : "layer_final"
# }
# model_transfer = create_feature_extractor(model_transfer, return_nodes=return_nodes)
#     # Freeze training for all 'features' layers
# for param in model_transfer.parameters():
#     param.requires_grad=False

# model_transfer.fc = nn.Linear(model_transfer.fc.in_features, nclass)