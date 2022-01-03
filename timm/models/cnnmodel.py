import sys
sys.path.append('../../../')
import torch.nn as nn
from torchstat import stat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

#model
class CNNModel(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel, self).__init__()
        self.n_classes = n_classes
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        self.convlayer1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.batch1
        )
        
        self.conv2 = nn.Conv2d(in_channels =32 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)
        self.convlayer2 = nn.Sequential(
            self.conv2, 
            self.relu2,
            self.batch2,
            self.maxpool1,
            self.conv1_drop
        )

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convlayer3 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.batch3
        )
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)
        self.convlayer4 = nn.Sequential(
            self.conv4,
            self.relu4,
            self.batch4,
            self.maxpool2,
            self.conv2_drop
        )

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(64*16,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,n_classes)
        self.fclayer = nn.Sequential(
            self.fc1,
            self.fc1_relu,
            self.dp1,
            self.fc2
        )
                
    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fclayer(out)

        return out

class CNNModel_split1(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel_split1, self).__init__()
        self.n_classes = n_classes
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        self.convlayer1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.batch1
        )
        
        self.conv2 = nn.Conv2d(in_channels =32 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)
        self.convlayer2 = nn.Sequential(
            self.conv2, 
            self.relu2,
            self.batch2,
            self.maxpool1,
            self.conv1_drop
        )

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convlayer3 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.batch3
        )
        
        self.conv4_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv4_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)
        self.convlayer4 = nn.Sequential(
            self.conv4_1,
            self.conv4_2,
            self.relu4,
            self.batch4,
            self.maxpool2,
            self.conv2_drop
        )

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(64*16,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,n_classes)
        self.fclayer = nn.Sequential(
            self.fc1,
            self.fc1_relu,
            self.dp1,
            self.fc2
        )
                
    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fclayer(out)

        return out

class CNNModel_split2(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel_split2, self).__init__()
        self.n_classes = n_classes
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        self.convlayer1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.batch1
        )
        
        self.conv2_1 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (1, 5), stride = 1, padding = 0 )
        self.conv2_2 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (5, 1), stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)
        self.convlayer2 = nn.Sequential(
            self.conv2_2,
            self.conv2_1, 
            self.relu2,
            self.batch2,
            self.maxpool1,
            self.conv1_drop
        )

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convlayer3 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.batch3
        )
        
        self.conv4_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv4_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)
        self.convlayer4 = nn.Sequential(
            self.conv4_2,
            self.conv4_1,
            self.relu4,
            self.batch4,
            self.maxpool2,
            self.conv2_drop
        )

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(64*16,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,n_classes)
        self.fclayer = nn.Sequential(
            self.fc1,
            self.fc1_relu,
            self.dp1,
            self.fc2
        )
                
    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fclayer(out)

        return out

class CNNModel_split3(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel_split3, self).__init__()
        self.n_classes = n_classes
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 5, stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        self.convlayer1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.batch1
        )
        
        self.conv2_1 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (1, 5), stride = 1, padding = 0 )
        self.conv2_2 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (5, 1), stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)
        self.convlayer2 = nn.Sequential(
            self.conv2_1,
            self.conv2_2, 
            self.relu2,
            self.batch2,
            self.maxpool1,
            self.conv1_drop
        )

        # Convolution layer 2
        self.conv3_0 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1)
        self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv3_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convlayer3 = nn.Sequential(
            self.conv3_0,
            self.conv3_1,
            self.conv3_2,
            self.relu3,
            self.batch3
        )
        
        self.conv4_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv4_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)
        self.convlayer4 = nn.Sequential(
            self.conv4_1,
            self.conv4_2,
            self.relu4,
            self.batch4,
            self.maxpool2,
            self.conv2_drop
        )

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(64*16,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,n_classes)
        self.fclayer = nn.Sequential(
            self.fc1,
            self.fc1_relu,
            self.dp1,
            self.fc2
        )
                
    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fclayer(out)

        return out

class CNNModel_split4(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel_split4, self).__init__()
        self.n_classes = n_classes
        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1_0 = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 1)
        self.conv1_1 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (1, 5), stride = 1, padding = 0 )
        self.conv1_2 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (5, 1), stride = 1, padding = 0 )
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        self.convlayer1 = nn.Sequential(
            self.conv1_0,
            self.conv1_1,
            self.conv1_2,
            self.relu1,
            self.batch1
        )
        
        self.conv2_1 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (1, 5), stride = 1, padding = 0 )
        self.conv2_2 = nn.Conv2d(in_channels = 32 , out_channels = 32, kernel_size = (5, 1), stride = 1, padding = 0 )
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_drop = nn.Dropout(0.25)
        self.convlayer2 = nn.Sequential(
            self.conv2_1,
            self.conv2_2, 
            self.relu2,
            self.batch2,
            self.maxpool1,
            self.conv1_drop
        )

        # Convolution layer 2
        self.conv3_0 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1)
        self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv3_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(64)
        self.convlayer3 = nn.Sequential(
            self.conv3_0,
            self.conv3_1,
            self.conv3_2,
            self.relu3,
            self.batch3
        )
        
        self.conv4_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3), stride = 1, padding = 0 )
        self.conv4_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 1), stride = 1, padding = 0 )
        self.relu4 = nn.ReLU()
        self.batch4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2_drop = nn.Dropout(0.25)
        self.convlayer4 = nn.Sequential(
            self.conv4_1,
            self.conv4_2,
            self.relu4,
            self.batch4,
            self.maxpool2,
            self.conv2_drop
        )

        # Fully-Connected layer 1
        
        self.fc1 = nn.Linear(64*16,256)
        self.fc1_relu = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        
        # Fully-Connected layer 2
        self.fc2 = nn.Linear(256,n_classes)
        self.fclayer = nn.Sequential(
            self.fc1,
            self.fc1_relu,
            self.dp1,
            self.fc2
        )
                
    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.convlayer1(x)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)

        #Flatten拉平操作
        out = out.view(out.size(0),-1)

        #FC layer的前向计算
        out = self.fclayer(out)

        return out

@register_model
def cnnmodel(pretrained=False, **kwargs):
    model = CNNModel(n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def cnnmodel_split1(pretrained=False, **kwargs):
    model = CNNModel_split1(n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def cnnmodel_split2(pretrained=False, **kwargs):
    model = CNNModel_split2(n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def cnnmodel_split3(pretrained=False, **kwargs):
    model = CNNModel_split3(n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def cnnmodel_split4(pretrained=False, **kwargs):
    model = CNNModel_split4(n_classes=10)
    model.default_cfg = _cfg
    return model