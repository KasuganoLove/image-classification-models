# image-classification-models

#### Introduction
image-classification-models:include resnet,vgg,alexnet,densnet

### Tensorboard
I use tensorboard to store the results.You can see the results by tesnsorboard at runs/{model}/train or simply set the logdir as ./runs

### Best Accurracy model
The models with best valid_acc are saved as runs/{model}/checkpoint.pth.See more information in source code.

#### Datasets
the datasets should be like cifar10
you should change the dir by change the source code at {model}-train.py

the codes is like:
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

### Example
you can simply use cmd to run:
    python {model}-train.py


