import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# 工具类
class modelTool:
    
    # 定义冻结模块，冻结模型所有参数权重
    def freeze_module(self, model):
        for param in model.parameters():
            param.requires_grad = False

    # 定义解冻模块，解冻模型所有参数权重
    def unfreeze_module(self, model):
        for param in model.parameters():
            param.requires_grad = True

    # 模型初始化
    def initialize_model(self, model_name, model, num_classes):
        # 选择合适的模型，不同模型的初始化方法稍微有点区别
        model_ft = model
        input_size = 0

        if model_name == "resnet":
            """ Resnet
            """
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                        nn.LogSoftmax(dim=1))
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def get_updatable_params(self, model):
        updatable_params_names = []
        updatable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                updatable_params_names.append(name)
                updatable_params.append(param)
        return updatable_params_names, updatable_params

    def train_model(self, model, dataloaders, filename, writer, num_epochs=300, epoch_shift=0, lr_start=1e-2):
        # 优化器设置
        optimizer = optim.Adam(model.parameters(), lr=lr_start)
        # 学习率每7个epoch衰减成原来的1/10
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # 余弦退火有序调整学习率
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # ReduceLROnPlateau（自适应调整学习率）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        # 损失函数
        criterion = nn.NLLLoss()

        # 是否用GPU训练
        train_on_gpu = torch.cuda.is_available()

        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        since = time.time()
        best_acc = 0

        model.to(device)

        val_acc_history = []
        train_acc_history = []
        train_losses = []
        valid_losses = []
        LRs = [optimizer.param_groups[0]['lr']]

        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            # 训练和验证
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # 训练
                else:
                    model.eval()  # 验证

                running_loss = 0.0
                running_corrects = 0

                # 把数据都取个遍
                for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 清零
                    optimizer.zero_grad()
                    # 只有训练的时候计算和更新梯度
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # 训练阶段更新权重
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 计算损失
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                time_elapsed = time.time() - since

                writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch + epoch_shift)  # tensorboard存储
                writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch + epoch_shift)  # tensorboard存储
                print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # 得到最好那次的模型
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    state = {
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state, filename)
                if phase == 'valid':
                    val_acc_history.append(epoch_acc.item())
                    valid_losses.append(epoch_loss)
                    scheduler.step(epoch_loss)
                if phase == 'train':
                    train_acc_history.append(epoch_acc.item())
                    train_losses.append(epoch_loss)

            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch + epoch_shift)  # tensorboard存储
            print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
            LRs.append(optimizer.param_groups[0]['lr'])
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 训练完后用最好的一次当做模型最终的结果
        model.load_state_dict(best_model_wts)
        return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs