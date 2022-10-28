# image-classification-models

## Introduction
image-classification-models, include :
    
    resnet50, resnet152, vgg, alexnet, densnet

## Initialization
You have to make some dirs, by :
    
    mkdir runs
    cd runs
    mkdir alexnet
    mkdir densenet
    mkdir resnet50
    mkdir resnet152
    mkdir vgg

## Tensorboard
I use tensorboard to store the results.

You can see the results by tesnsorboard at runs/{model}/train or simply set the logdir as ./runs:

    tensorboard --logdir="./runs"

## Best Accurracy model
The models which have the best valid_acc are saved as runs/{model}/checkpoint.pth.

See more information in source code.

## Datasets
the datasets should be like cifar10:

    ——dataset_name
        ——train
        ——valid


you should change the dir by change the source code at {model}-train.py

    data_dir = './scrapsteel/'

you can download our experiment dataset through :

    git clone https://github.com/flashszn/ScrapSteelDataset

## Example
you can simply use cmd to run :
    
    python {model}-train.py

## Freeze Train
Dataset too small ? 

You can freeze all parameters and only train the fully connected layer by :    

    python {model}-train.py --freeze

## Clean
So many train logs,what a messy !
Use clean.py to clean up ./runs or ./runs/{model} !
    
    python clean.py
    python clean.py --model alexnet

## Device
You can choose your cuda device by using --device, for example:

    python resnet50-train.py --device 3



