# Rice-Leaf-Disease-Detection

```
Our aim is to classify the given rice leaf image into one of the given four categories.
```
Status : Inprogress
---------------------
## Results Comparison

For comparison, we use the results from the paper Sethy et al.([link](https://www.sciencedirect.com/science/article/abs/pii/S0168169919326997?via=ihub)) with their best model of using ResNet for feature extraction and SVM on top of it for classification.
Our models include Inception+attention, ResNet + Attention and VGG + Attention on the same dataset

| Method  | ResNet + SVM (original paper) | Inception + Attention | ResNet + Attention	|  VGG + Attention |
| ------------- | ------------- | ------------ |------------|-------------|
| Bacterial blight  | 76  |	37 | 56 | 85.3 |
| Blast  | 51 |	35 | 41 | 89.5 |
| Brown spot  | 83  | 34 | 46 | 94.8 |
| Tungros  | 52  | 6 |	8 | 84.1 |

## Dataset
-----------
We use the rice leaf dataset obtained from Sethy et al. [Dataset](https://data.mendeley.com/datasets/fwcj7stb8r/1)
Dataset consists of 4 classes {Bacterial blight, Blast, Brown spot, Tungro.}

## Installation
--------------
##### Requirements
```
pytorch >= 1.2.0
pandas
numpy
tqdm
scikit-learn
torchvision
```

<!--#### Installing without GPU:-->
<!--```-->
<!--pip3 install requirements.txt-->
<!--```-->
**To install and use with GPU, cuda toolkit along with drivers need to be installed.
And set *use_cuda = 1* in training/testing codes.**

## Testing and Training using ResNet-101 
This is for training 30 times with random sampling of data, where each of the 30 runs will be run for 50 epochs. Testing is done for every run once and the models and test results are stored in the same directory
```
git clone https://github.com/sraone-96/Rice-Leaf-Disease-Detection.git
cd Rice-Leaf-Disease-Detection
python3 dm_project_test_included.py
```
**NOTE** : Make sure to have the corresponding image data in the same folder where you are running the code from. And name the folder as `Rice Leaf Disease Images`

