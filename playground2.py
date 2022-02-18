import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

print("train_data:", train_dataset.data.size())
print("train_labels:", train_dataset.targets.size())
print("test_data:", test_dataset.data.size())
print("test_labels:", test_dataset.targets.size())

# load_data
batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

print("batch_size:", train_loader.batch_size)
print("load_train_data:", train_loader.dataset.data.shape)
print("load_train_labels:", train_loader.dataset.targets.shape)
print(len(train_loader))

# i=0 
# plt.imshow(train_dataset.data[i])
# plt.title(train_dataset.targets[i]) 
# plt.show()


class CNN(nn.Module):
    def __init__(self, dim, n_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, dim, kernel_size=3, padding="same")
        self.actlayer = nn.ReLU() 
        self.bn1 = nn.BatchNorm2d(dim)
        self.hidden1 = nn.Conv2d(dim, dim, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(dim)
        self.hidden2 = nn.Conv2d(dim, dim, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.actlayer(x)
        x = self.bn1(x)
        x = self.hidden1(x)
        x = self.actlayer(x)
        x = self.bn2(x)
        x = self.hidden2(x)
        x = self.actlayer(x)
        x = self.bn3(x)
        x = self.classifier(x)
        return x 

if __name__ == "__main__":
    #定义损失函数和优化器
    net = CNN(dim=64) #实例化网络 dim是多少维，是可调参数
    net = net.cuda()
    print(net)
    criterion = nn.CrossEntropyLoss() #定义损失函数
    learning_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) #定义优化器


    #训练过程
    #训练多少轮
    num_epoches = 100
    for epoch in range(num_epoches):
        print("current epoch = {}".format(epoch))   
        total = 0
        correct = 0
        #遍历60000张图
        for i, (images,labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = criterion(outputs, labels)  # calculate loss
            optimizer.zero_grad()  # clear net state before backward optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()       #反向传播计算出每个参数的梯度
            optimizer.step()   # update parameters 计算出来的这个梯度更新的每个参数
            _,predicts = torch.max(outputs.data, 1) #其中这个 1代表行，0的话代表列。 不加_,返回的是一行中最大的数。加_,则返回一行中最大数的位置。
            total += labels.size(0)
            correct += (predicts == labels).sum()

            if i%10 == 0:
                print(i)
                print("current loss = %.5f" %loss.item())
                print("Accuracy = %.2f" %(100*correct/total))
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (images,labels) in enumerate(test_loader):
                images = images.cuda()
                labels = labels.cuda()

                outputs = net(images)
                loss = criterion(outputs, labels)  # calculate loss
                _,predicts = torch.max(outputs.data, 1) #其中这个 1代表行，0的话代表列。 不加_,返回的是一行中最大的数。加_,则返回一行中最大数的位置。
                total += labels.size(0)
                correct += (predicts == labels).sum()
        print("test Accuracy = %.2f" %(100*correct/total))
    print("finished training")

